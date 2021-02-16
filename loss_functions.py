import tensorflow as tf
from tensorflow.keras.backend import epsilon

eps = epsilon()

def negative_cos(pred, true):
    return 1 - tf.math.divide_no_nan(tf.reduce_sum(pred * true, axis = 1),
            tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1))

def angle(pred, true):
    return tf.math.acos(
        tf.clip_by_value(
            tf.math.divide_no_nan(tf.reduce_sum(pred * true, axis = 1),
            tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1)),
            -1., 1.)
        )

def loss_func_negative_cos(y_reco, y_true, return_from = False):
    # Energy loss
    loss_energy    = tf.reduce_mean(
        tf.abs(
            tf.subtract(
                y_reco[:,0], y_true[:,0]
                )
            )
        )
    loss           = loss_energy

    # Position loss
    loss_dist  = tf.reduce_mean(
        tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        y_reco[:, 1:4], y_true[:, 1:4]
                    )
                ), axis = 1
            )
        )
    )
    loss       += loss_dist

    loss_angle = tf.reduce_mean(negative_cos(y_reco[:, 4:], y_true[:,4:]))
    loss       += loss_angle
    # loss      += tf.reduce_mean(angle(y_reco[:, 4:], y_true[:, 4:]))
    if return_from:
        return float(loss_energy), float(loss_dist), float(loss_angle)
    else:
        return loss


def loss_func_linear_angle(y_reco, y_true, return_from = False):
    # Energy loss
    loss_energy    = tf.reduce_mean(
        tf.abs(
            tf.subtract(
                y_reco[:,0], y_true[:,0]
                )
            )
        )
    loss           = loss_energy

    # Position loss
    loss_dist  = tf.reduce_mean(
        tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    tf.subtract(
                        y_reco[:, 1:4], y_true[:, 1:4]
                    )
                ), axis = 1
            )
        )
    )
    loss       += loss_dist

    cos_angle = tf.math.divide_no_nan(tf.reduce_sum(y_reco[:, 4:] * y_true[:, 4:], axis = 1),
            tf.math.reduce_euclidean_norm(y_reco[:, 4:], axis = 1) * tf.math.reduce_euclidean_norm(y_true[:, 4:],  axis = 1))

    cos_angle -= tf.math.sign(cos_angle) * 1e-6
    loss_angle = tf.reduce_mean(tf.math.acos(cos_angle))

    loss       += loss_angle
    
    
    if return_from:
        return float(loss_energy), float(loss_dist), float(loss_angle)
    else:
        return loss