import tensorflow as tf
from util import *


def crossgrad_latent_fn(features, labels, mode, params):
    # Some convolutions
    features = tf.layers.conv2d(features, 32, 3, data_format=params['data_format'], activation=tf.nn.relu)
    features = tf.layers.conv2d(features, 32, 3, data_format=params['data_format'], activation=tf.nn.relu)
    features = tf.layers.max_pooling2d(features, 2, 1, data_format=params['data_format'])
    features = tf.layers.conv2d(features, 128, 3, data_format=params['data_format'], activation=tf.nn.relu)
    features = tf.layers.conv2d(features, 128, 3, data_format=params['data_format'], activation=tf.nn.relu)
    features = tf.layers.max_pooling2d(features, 2, 1, data_format=params['data_format'])
    features = tf.layers.conv2d(features, 256, 3, data_format=params['data_format'], activation=tf.nn.relu)
    features = tf.layers.conv2d(features, 256, 3, data_format=params['data_format'], activation=tf.nn.relu)
    
    # Down to [batch, 1, 1, 256] -> [batch, 1, 1, 256]
    size = features.shape.as_list()[2]
    features = tf.layers.max_pooling2d(features, size, 1, data_format=params['data_format'])

    # Target dimensions [batch, 1, 1, target] -> [batch, target]
    features = tf.layers.conv2d(features, params['latent_space_dimensions'], 1, data_format=params['data_format'])
    features = tf.layers.flatten(features)
    
    return features

def crossgrad_domain_fn(features, labels, mode, params):
    # Compute logits
    features = tf.layers.dense(features, params['latent_space_dimensions'], activation=tf.nn.relu)
    logits = tf.layers.dense(features, params['num_domain'])
    
    # Final classification
    softmax = tf.nn.softmax(logits)
    domain = tf.argmax(softmax, axis=-1)
    
    # During inference we don't have labels
    loss = 0
    if labels is not None:
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, params['num_domain']),
            logits=logits
        )
    
    return domain, tf.reduce_mean(loss)


def crossgrad(mode, label_fn, x, domain, labels, epsilon_d, epsilon_l, alpha_d, alpha_l, latent_fn=None, domain_fn=None, latent_domain_fn=None, latent_space_dimensions=100, params=None):
    # Some default parameters
    if params is None: params = {}
    
    if 'data_format' not in params or params['data_format'] is None:
        params['data_format'] = detect_data_format()
        
    params['latent_space_dimensions'] = latent_space_dimensions
    
    # Given functions or default
    latent_fn = latent_fn or crossgrad_latent_fn
    domain_fn = domain_fn or crossgrad_domain_fn
    
    # To expected data format
    x = to_data_format(x, 'channels_last', params['data_format'])
    
    def forward(x_l, x_d, d, labels, mode, params):
        if latent_domain_fn is not None:
            latent, domain, loss_domain = latent_domain_fn(features=x_l, labels=d, mode=mode, params=params)
        else:
            latent = latent_fn(features=x_l, labels=None, mode=mode, params=params)
            domain, loss_domain = domain_fn(features=latent, labels=d, mode=mode, params=params)
        latent = tf.stop_gradient(latent)
        outputs, loss_label = label_fn(features=x_d, latent=latent, labels=labels, mode=mode, params=params)

        return ((domain, loss_domain), (outputs, loss_label))
    
    # First forward pass
    with tf.variable_scope('crossgrad', reuse=False):
        ((d1, d1_loss), (l1, l1_loss)) = forward(x, x, domain, labels, mode, params)
        
    if mode != tf.estimator.ModeKeys.PREDICT:    
        grad_label_x = tf.stop_gradient(tf.gradients(l1_loss, x)[0])
        grad_domain_x = tf.stop_gradient(tf.gradients(d1_loss, x)[0])

        grad_label_x = tf.clip_by_value(grad_label_x, -0.1, 0.1)
        grad_domain_x = tf.clip_by_value(grad_domain_x, -0.1, 0.1)

        if epsilon_l == 0:
            grad_label_x = 0

        if epsilon_d == 0:
            grad_domain_x = 0

        # Second pass
        with tf.variable_scope('crossgrad', reuse=True):
            x_d = x + epsilon_d * grad_domain_x
            x_l = x + epsilon_l * grad_label_x
            ((d2, d2_loss), (l2, l2_loss)) = forward(x_l, x_d, domain, labels, mode, params)

        l_total_loss = (1 - alpha_l) * l1_loss + alpha_l * l2_loss
        d_total_loss = (1 - alpha_d) * d1_loss + alpha_d * d2_loss
        predictions = {
            'd1': d1,
            'd2': d2,
            'l1': l1,
            'l2': l2
        }

        if params.get('summaries'):
            tf.summary.scalar('loss/domain/1', d1_loss)
            tf.summary.scalar('loss/labels/1', l1_loss)

            tf.summary.scalar('loss/domain/2', d2_loss)
            tf.summary.scalar('loss/labels/2', l2_loss)    

            tf.summary.image('x', to_data_format(x, params['data_format'], 'channels_last'))
            tf.summary.image('x_d', to_data_format(x_d, params['data_format'], 'channels_last'))
            tf.summary.image('x_l', to_data_format(x_l, params['data_format'], 'channels_last'))

            if epsilon_d > 0:
                tf.summary.image('grad_d', to_data_format(epsilon_d * grad_domain_x, params['data_format'], 'channels_last'))
                tf.summary.histogram('grad_d', to_data_format(epsilon_d * grad_domain_x, params['data_format'], 'channels_last'))

            if epsilon_l > 0:
                tf.summary.image('grad_l', to_data_format(epsilon_l * grad_label_x, params['data_format'], 'channels_last'))
                tf.summary.histogram('grad_l', to_data_format(epsilon_l * grad_label_x, params['data_format'], 'channels_last'))
    else:
        predictions = {
            'd1': d1,
            'l1': l1
        }
        l_total_loss = d_total_loss = 0
    
    return predictions, l_total_loss + d_total_loss
