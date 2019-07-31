"""Test if the pre-trained weights can be loaded correctly.

"""
import os
import pathlib

import keras
import numpy
import sklearn.cluster
import sklearn.metrics.pairwise
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

def main(weight_path, wav_dir, slice_len, step_size, eps, outpath):
    model = DiarizerModel(weight_path, eps)
    model.diarize(wav_dir, slice_len, step_size, outpath)

class DiarizerModel:

    def __init__(self, weight_path, cluster_threshold=0.9):
        self.net = model.vggvox_resnet2d_icassp(
            input_dim=(257, None, 1),
            num_class=5994,
            mode="eval",
            args=DefaultEvalArgs()
        )
        self.net.load_weights(weight_path)

        self.cluster_threshold = cluster_threshold

    def diarize(self, wav_dir, slice_len=400, step_size=32, outpath="diarization-results.npy"):
        fpaths = pathlib.Path(wav_dir).rglob("*.wav")
        fpaths = list(tqdm.tqdm(fpaths, desc="Collecting files", ncols=80))

        decoder = []
        encoded = []

        for fpath in tqdm.tqdm(fpaths, ncols=80, desc="Processing spectrograms"):
            dataloader = create_dataloader(fpath, slice_len, step_size)
            embeddings = embed_slices(dataloader, self.net)
            clusterer = sklearn.cluster.KMeans(n_clusters=3)
            clusterer.fit(embeddings)
            c1, c2, c3 = clusterer.cluster_centers_
            dist_c12 = cosine_sim(c1, c2)
            dist_c13 = cosine_sim(c1, c3)
            dist_c23 = cosine_sim(c2, c3)
            labels = clusterer.labels_
            label_len = len(labels)
            key = int(os.path.basename(fpath)[:-4])
            decoder.append([key, label_len, dist_c12, dist_c13, dist_c23])
            encoded.append(labels)

        out = [numpy.array(decoder), numpy.concatenate(encoded, axis=0)]
        numpy.save(outpath, out)

def cosine_sim(v1, v2):
    return (v1 * v2).sum() / numpy.linalg.norm(v1) / numpy.linalg.norm(v2)

def create_dataloader(fpath, slice_len, step_size):
    spec = load_data(fpath)
    dset = dataset.Dataset(spec, slice_len, step_size)
    return dataset.DataLoader(dset, batch_size=8)

def embed_slices(dataloader, net):
    out = []
    for i in range(len(dataloader)):
        batch = dataloader[i]
        pred = net.predict(batch)
        out.append(pred)
    return numpy.concatenate(out, axis=0)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--wave_dir", required=True)
    parser.add_argument("--slice_len", type=int, required=True)
    parser.add_argument("--step_size", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--outpath", required=True)

    args = parser.parse_args()

    main(
        args.weight_path, args.wave_dir, args.slice_len,
        args.step_size, args.eps, args.outpath
    )
