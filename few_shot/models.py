import paddle
from paddle.vision.models import resnet152


class Flatten(paddle.nn.Layer):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return input.view(input.shape[0], -1)


class GlobalMaxPool1d(paddle.nn.Layer):
    """Performs global max pooling over the entire length of a batched 1D tensor
    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return paddle.nn.functional.max_pool1d(
            x=input, kernel_size=input.shape[2:]
        ).view(-1, input.shape[1])


class GlobalAvgPool2d(paddle.nn.Layer):
    """Performs global average pooling over the entire height and width of a batched 2D tensor

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return paddle.nn.functional.avg_pool2d(
            kernel_size=input.shape[2:], x=input, exclusive=False
        ).view(-1, input.shape[1])


def conv_block(in_channels: int, out_channels: int) -> paddle.nn.Layer:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return paddle.nn.Sequential(
        paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        ),
        paddle.nn.BatchNorm2D(num_features=out_channels),
        paddle.nn.ReLU(),
        paddle.nn.MaxPool2D(kernel_size=2, stride=2),
    )


def functional_conv_block(
    x: paddle.Tensor,
    weights: paddle.Tensor,
    biases: paddle.Tensor,
    bn_weights,
    bn_biases,
) -> paddle.Tensor:
    """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.

    # Arguments:
        x: Input Tensor for the conv block
        weights: Weights for the convolutional block
        biases: Biases for the convolutional block
        bn_weights:
        bn_biases:
    """
    x = paddle.nn.functional.conv2d(x=x, weight=weights, bias=biases, padding=1)
    x = paddle.nn.functional.batch_norm(
        x=x,
        running_mean=None,
        running_var=None,
        weight=bn_weights,
        bias=bn_biases,
        training=True,
    )
    x = paddle.nn.functional.relu(x=x)
    x = paddle.nn.functional.max_pool2d(x=x, kernel_size=2, stride=2)
    return x


def get_few_shot_encoder(num_input_channels=1) -> paddle.nn.Layer:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. fashionNet = 3
    """
    return paddle.nn.Sequential(
        conv_block(num_input_channels, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        Flatten(),
    )


def resnet_pretrained(num_input_channels=1) -> paddle.nn.Layer:
    """Creates a ResNet pretrained metwork as Prototypical Networks"""
    model = resnet152(pretrained=False)
    modules = list(model.children())[:-2]
    return paddle.nn.Sequential(*modules)


class FewShotClassifier(paddle.nn.Layer):
    def __init__(self, num_input_channels: int, k_way: int, final_layer_size: int = 64):
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(FewShotClassifier, self).__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.logits = paddle.nn.Linear(in_features=final_layer_size, out_features=k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape((x.shape[0], -1))
        return self.logits(x)

    def functional_forward(self, x, weights):
        """Applies the same forward pass using PyTorch functional operators using a specified set of weights."""
        for block in [1, 2, 3, 4]:
            x = functional_conv_block(
                x,
                weights["conv{}.0.weight".format(block)],
                weights["conv{}.0.bias".format(block)],
                weights.get("conv{}.1.weight".format(block)),
                weights.get("conv{}.1.bias".format(block)),
            )
        x = x.view(x.shape[0], -1)
        x = paddle.nn.functional.linear(
            weight=weights["logits.weight"].T, bias=weights["logits.bias"], x=x
        )
        return x
