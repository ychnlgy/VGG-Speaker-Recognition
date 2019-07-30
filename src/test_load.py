"""Test if the pre-trained weights can be loaded correctly.

"""
import pathlib

import keras
import numpy
import tqdm

import dataset
import model
import utils

class DefaultEvalArgs:

    def __init__(self):
        self.net = "resnet34s"
        self.loss = "softmax"
        self.vlad_cluster = 8
        self.ghost_cluster = 2
        self.bottleneck_dim = 512
        self.aggregation_mode = "gvlad"

def load_data(fpath):
    spec = utils.load_data(
        fpath,
        win_length=400,  # TODO: make shorter?
        sr=16000,
        hop_length=160,
        n_fft=512,
        spec_len=250,
        mode="eval"
    )
    return numpy.expand_dims(
        numpy.expand_dims(spec, 0),
        -1
    )

def main(weight_path, wav_dir, slice_len, step_size):

    net = model.vggvox_resnet2d_icassp(
        input_dim=(257, None, 1),
        num_class=5994,
        mode="eval",
        args=DefaultEvalArgs()
    )

    net.load_weights(weight_path)

    fpaths = pathlib.Path(wav_dir).rglob("*.wav")
    fpaths = list(tqdm.tqdm(fpaths, desc="Collecting files", ncols=80))

    for fpath in tqdm.tqdm(fpaths, ncols=80, desc="Processing spectrograms"):
        dataloader = create_dataloader(fpath, slice_len, step_size)
        embeddings = embed_slices(dataloader, net)
        print(embeddings.shape)
        input("Temp stop")
        # K-means or DBSCAN
        # Return clusterer.labels_
        raise NotImplementedError

def create_dataloader(fpath, slice_len, step_size):
    spec = load_data(fpath)
    dset = dataset.Dataset(spec, slice_len, step_size)
    return dataset.DataLoader(dset, batch_size=8)

def embed_slices(dataloader, net):
    out = []
    for i in range(len(dataloader)):
        batch = dataloader[i]
        pred = net.predict(batch)
        out.append(batch)
    return numpy.concatenate(out, axis=0)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--wave_dir", required=True)
    parser.add_argument("--slice_len", type=int, required=True)
    parser.add_argument("--step_size", type=int, required=True)

    args = parser.parse_args()

    main(args.weight_path, args.wave_dir, args.slice_len, args.step_size)
