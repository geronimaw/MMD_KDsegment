import torch
import tensorflow as tf
from mmd_torch import MMD as MMD_torch
from mmd_tf import MMD as MMD_tf


if __name__ == '__main__':
    a = torch.tensor([[1., 2.], [2, 3]])
    b = torch.tensor([[4., 6.], [8, 0]])
    c = torch.tensor([[1., 1.], [2, 3]])

    ab = MMD_torch(a, b, "multiscale")
    ac = MMD_torch(a, c, "multiscale")
    aa = MMD_torch(a, a, "multiscale")

    print("MMD_torch(a,b) = ", ab)
    print("MMD_torch(a,c) = ", ac)
    print("MMD_torch(a,a) = ", aa)

    a = tf.constant([[1., 2.], [2, 3]], dtype=tf.float32)
    b = tf.constant([[4., 6.], [8., 0.]], dtype=tf.float32)
    c = tf.constant([[1., 1.], [2, 3]])

    ab = MMD_tf(a, b, "multiscale")
    ac = MMD_tf(a, c, "multiscale")
    aa = MMD_tf(a, a, "multiscale")

    print("MMD_tf(a,b) = ", ab)
    print("MMD_tf(a,c) = ", ac)
    print("MMD_tf(a,a) = ", aa)
