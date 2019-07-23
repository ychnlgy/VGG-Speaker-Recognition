"""Test if the pre-trained weights can be loaded correctly.

"""

import model

class DefaultEvalArgs:

    def __init__(self):
        self.net = "resnet34s"
        self.loss = "softmax"
        self.vlad_cluster = 8
        self.ghost_cluster = 2
        self.bottleneck_dim = 512
        self.aggregation_mode = "gvlad"

def main(weight_path):

    net = model.vggvox_resnet2d_icassp(
        input_dim=(257, None, 1),
        num_class=5994,
        mode="eval",
        args=DefaultEvalArgs()
    )

    net.load_weights(weight_path)

    print(net.count_params())

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path", required=True)

    args = parser.parse_args()

    main(args.weight_path)
