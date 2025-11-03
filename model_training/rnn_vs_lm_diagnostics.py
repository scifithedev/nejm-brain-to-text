import os
import time
import argparse
import numpy as np
import editdistance
from tqdm import tqdm

from evaluate_model_helpers import *
from rnn_model import GRUDecoder
import torch
from omegaconf import OmegaConf


def load_model(model_path, device):
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
    model = GRUDecoder(
        neural_dim = model_args['model']['n_input_features'],
        n_units = model_args['model']['n_units'], 
        n_days = len(model_args['dataset']['sessions']),
        n_classes = model_args['dataset']['n_classes'],
        rnn_dropout = model_args['model']['rnn_dropout'],
        input_dropout = model_args['model']['input_network']['input_layer_dropout'],
        n_layers = model_args['model']['n_layers'],
        patch_size = model_args['model']['patch_size'],
        patch_stride = model_args['model']['patch_stride'],
    )
    checkpoint = torch.load(
        os.path.join(model_path, 'checkpoint/best_checkpoint'),
        map_location=device,
        weights_only=False,
    )
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace("module.", "")] = checkpoint['model_state_dict'].pop(key)
        checkpoint['model_state_dict'][key.replace("_orig_mod.", "")] = checkpoint['model_state_dict'].pop(key)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, model_args


def greedy_decode_logits(logits):
    # logits shape: [T, C]
    pred_seq = np.argmax(logits, axis=-1)
    pred_seq = [int(p) for p in pred_seq if p != 0]
    pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
    pred_phonemes = [LOGIT_TO_PHONEME[p] for p in pred_seq]
    return pred_phonemes


def run_rnn_only(model_path, data_dir, csv_path, max_trials, device):
    # load csv
    import pandas as pd
    b2txt_csv_df = pd.read_csv(csv_path)

    model, model_args = load_model(model_path, device)

    # load all sessions
    test_data = {}
    total = 0
    for session in model_args['dataset']['sessions']:
        files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
        if f'data_val.hdf5' in files:
            eval_file = os.path.join(data_dir, session, f'data_val.hdf5')
            data = load_h5py_file(eval_file, b2txt_csv_df)
            test_data[session] = data
            total += len(data['neural_features'])

    pbar_total = min(total, max_trials) if max_trials else total
    processed = 0

    total_true_len = 0
    total_ed = 0
    examples = []

    with tqdm(total=pbar_total, desc='RNN-only decode', unit='trial') as pbar:
        stop = False
        for session, data in test_data.items():
            input_layer = model_args['dataset']['sessions'].index(session)
            for trial in range(len(data['neural_features'])):
                x = np.expand_dims(data['neural_features'][trial], axis=0)
                tensor_dtype = torch.bfloat16 if (device.type != 'cpu' and torch.cuda.is_available()) else torch.float32
                x_t = torch.tensor(x, device=device, dtype=tensor_dtype)
                logits = runSingleDecodingStep(x_t, input_layer, model, model_args, device)
                # rearrange for logits order expected by helpers (keep batch dim)
                logits = rearrange_speech_logits_pt(logits)[0]
                pred_phonemes = greedy_decode_logits(logits)

                # ground truth phonemes
                true_ids = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
                true_phonemes = [LOGIT_TO_PHONEME[int(p)] for p in true_ids]

                ed = editdistance.eval(true_phonemes, pred_phonemes)
                total_true_len += len(true_phonemes)
                total_ed += ed

                if len(examples) < 20:
                    examples.append({
                        'session': session,
                        'block': data['block_num'][trial],
                        'trial': data['trial_num'][trial],
                        'true_sentence': data['sentence_label'][trial],
                        'true_phonemes': ' '.join(true_phonemes),
                        'rnn_phonemes': ' '.join(pred_phonemes),
                    })

                processed += 1
                pbar.update(1)
                if max_trials is not None and processed >= max_trials:
                    stop = True
                    break
            if stop:
                break
    per = (total_ed / total_true_len) if total_true_len > 0 else None
    return per, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_trials', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='data/t15_pretrained_rnn_baseline')
    parser.add_argument('--data_dir', type=str, default='data/hdf5_data_final')
    parser.add_argument('--csv_path', type=str, default='data/t15_copyTaskData_description.csv')
    parser.add_argument('--gpu_number', type=int, default=-1)
    args = parser.parse_args()

    device = torch.device('cpu') if args.gpu_number < 0 or not torch.cuda.is_available() else torch.device(f'cuda:{args.gpu_number}')

    print('Running RNN-only diagnostic...')
    per, examples = run_rnn_only(args.model_path, args.data_dir, args.csv_path, args.max_trials, device)

    print(f'RNN-only PER over {args.max_trials} trials: {100*per:.2f}% (phoneme-level)')
    print('\nExamples (first 10):')
    for ex in examples[:10]:
        print(f"Session {ex['session']} Block {ex['block']} Trial {ex['trial']}")
        print('True sentence:', ex['true_sentence'])
        print('True phonemes:', ex['true_phonemes'])
        print('RNN phonemes: ', ex['rnn_phonemes'])
        print()


if __name__ == '__main__':
    main()
