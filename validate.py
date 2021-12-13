import pickle
import time
import argparse

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from PIL import Image
import random

from model import EncoderCNN, AttnDecoderRNN
from prepro import Vocabulary
from data_loader import get_loader
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from utils import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='coco2014')
parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
parser.add_argument('--test_image', type=str, default=None, help='path for testing image')
parser.add_argument('--image_dir', type=str, default='data/train2014_resized', help='directory for resized images')
parser.add_argument('--image_dir_val', type=str, default='data/val2014_resized', help='directory for resized images')
parser.add_argument('--image_dir_test', type=str, default='data/test2014')
parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                    help='path for train annotation json file')
parser.add_argument('--caption_path_val', type=str, default='data/annotations/captions_val2014.json',
                    help='path for val annotation json file')
parser.add_argument('--log_step', type=int, default=100, help='step size for printing log info')
parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

# Model parameters
parser.add_argument('--embed_dim', type=int, default=512, help='dimension of word embedding vectors')
parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers')
parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder rnn')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--epochs_since_improvement', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--encoder_lr', type=float, default=1e-4)
parser.add_argument('--decoder_lr', type=float, default=4e-4)
parser.add_argument('--checkpoint', type=str, default=None, help='path for checkpoints')
parser.add_argument('--grad_clip', type=float, default=5.)
parser.add_argument('--alpha_c', type=float, default=1.)
parser.add_argument('--best_bleu4', type=float, default=0.)
parser.add_argument('--fine_tune_encoder', type=bool, default='False', help='fine-tune encoder')

args = parser.parse_args()
print(args)


def main(args):
    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    if args.checkpoint is None:
        raise Exception("Please give the checkpoint file!")
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        print("Epoch:", checkpoint['epoch'])
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        encoder = checkpoint['encoder']
        encoder.max_sentence_length = 50

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Build data loader
    val_loader = get_loader(args.image_dir_val, args.caption_path_val, vocab,
                            transform, int(args.batch_size / 2),
                            num_workers=args.num_workers,
                            val=True
                           )
    # Use this to get bleu score on validation set
    # bleu_ave, bleu1, bleu2, bleu3, bleu4 = validate(val_loader=val_loader,
    #                        encoder=encoder,
    #                        decoder=decoder,
    #                        criterion=criterion)
    # Use this to predict a caption on a given image
    predict(val_loader, vocab, encoder, decoder, image_path=args.test_image)
    print(f"Average Bleu: {bleu_ave}")
    print(f"Bleu1: {bleu1}")
    print(f"Bleu2: {bleu2}")
    print(f"Bleu3: {bleu3}")
    print(f"Bleu4: {bleu4}")



def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        with torch.no_grad():
            # Forward prop.
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % args.log_step == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            word_map = val_loader.dataset.vocab
            for j in range(len(allcaps)):
                img_caps = allcaps[j]
                img_captions = list(
                    map(lambda c: [w.item() for w in c if w.item() not in {word_map('<start>'), word_map('<pad>')}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

            # Calculate BLEU-4 scores
            if i % 100 == 0:
                vocab = val_loader.dataset.vocab
                meteor = []
                for (ref, hypo) in zip(references, hypotheses):
                    ref_text = []
                    for ref_single in ref:
                        ref_text.append([vocab.idx2word[int(token)] for token in ref_single])
                    hypo_text = [vocab.idx2word[int(token)] for token in hypo]
                    meteor.append(meteor_score(ref_text, hypo_text))
                meteor = sum(meteor) / len(meteor)
                bleu_ave = corpus_bleu(references, hypotheses)
                bleu_1 = corpus_bleu(references, hypotheses, weights=(1,0,0,0))
                bleu_2 = corpus_bleu(references, hypotheses, weights=(0,1,0,0))
                bleu_3 = corpus_bleu(references, hypotheses, weights=(0,0,1,0))
                bleu_4 = corpus_bleu(references, hypotheses, weights=(0,0,0,1))
                print(
                    '\n * bleu_ave: {}, bleu1: {}, bleu2: {}, bleu3: {}, bleu4: {}, metero: {}\n'.format(
                    bleu_ave,
                    bleu_1,
                    bleu_2,
                    bleu_3,
                    bleu_4,
                    meteor
                    ))

    return bleu_ave, bleu_1, bleu_2, bleu_3, bleu_4

def predict(val_loader, vocab, encoder, decoder, image_path=None):
    if image_path:
        image_ori = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.CenterCrop(min(image_ori.width, image_ori.height)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        image = transform(image_ori)
        caption_ref = None
    else:
        id = random.randint(0, len(val_loader))
        image, caption_ref, _ = val_loader.dataset.__getitem__(id)
        image_ori = image
    decoder.eval()  # eval mode (no dropout or batchnorm)
    encoder.eval()
    image = image.unsqueeze(0)
    image = image.to(device)
    feature = encoder(image)
    caption_pred, encoded_captions, decode_lengths, alphas = decoder(feature)
    caption_pred = caption_pred.squeeze(0)
    caption_pred_text = []
    for i in range(len(caption_pred)):
        token_id = caption_pred[i].argmax().item()
        word = vocab.idx2word[token_id]
        caption_pred_text.append(word)
        if word == '<end>':
            break
    try:
        plt.imshow(image_ori.permute(1,2,0))
    except:
        image_ori.show()
    print(" ".join(caption_pred_text))

if __name__ == '__main__':
    main(args)
