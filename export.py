import paddle
from few_shot.models import FewShotClassifier
if __name__ == '__main__':
    model = FewShotClassifier(3, 3, 1600)
    x = paddle.randn(shape=[1, 3, 80, 80])
    try:
        x = paddle.static.InputSpec.from_tensor(x)
        paddle.jit.save(model, input_spec=(x,), path="./model")
        print('[JIT] paddle.jit.save successed.')
        exit(0)
    except Exception as e:
        print('[JIT] paddle.jit.save failed.')
        raise e
