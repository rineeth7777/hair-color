import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
import data
import random
import module
import tensorflow_datasets as tfds



# ==============================================================================
# =                                   param                                    =
# ==============================================================================

'''py.arg('--dataset', default='summer2winter_yosemite')
py.arg('--datasets_dir', default='dataset')
py.arg('--load_size', type=int, default=256)  # load image to this size
py.arg('--crop_size', type=int, default=256)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()'''
adataset='hair'
aload_size=256
acrop_size=256
abatch_size=1
apool_size=50
aadversarial_loss_mode='lsgan'
aepochs=200
agradient_penalty_weight=10.0
alr=0.0002
aepoch_decay=100
abeta_1=0.5
acycle_loss_weight=10.0
aidentity_loss_weight=0.1
agradient_penalty_mode='none'



# output_dir
output_dir = py.join('output', adataset)
py.mkdir(output_dir)

# save settings
#py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

def createdata(l):
    n=len(l)
    lis=[]
    for k in range(n):
        s=l[k].split('\\')[-1]
        s2='data\\tests\\0.1_color\\'+s
        lis.append(s2)
    return lis

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

B_imgs = py.glob(py.join('data\\tests', 'trainsmall'), '*.jpg')
B_color=createdata(B_imgs)
B_set=data.make_dataset( B_imgs,B_color,abatch_size, aload_size, acrop_size, training=False,work=0)

B_lis=list(tfds.as_numpy(B_set))
B_list=[]
for b in B_lis:
    m,=b
    B_list.append(tf.convert_to_tensor(m))

B_length=len(B_list)
len_dataset=B_length
#print('b list\n',B_list[0][0].squeeze().shape)


A2B_pool = data.ItemPool(apool_size)
#B2A_pool = data.ItemPool(a.pool_size)

print('session starts \n')
print('B',B_length)



# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G = module.ResnetGenerator(input_shape=(acrop_size, acrop_size, 6))

D= module.ConvDiscriminator(input_shape=(acrop_size, acrop_size, 6))
#print('generator')
#print(G.summary())

#print('discriminator')
#print(D.summary())


d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(aadversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(alr, aepochs * len_dataset, aepoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(alr, aepochs * len_dataset, aepoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=abeta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=abeta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A,B):
    with tf.GradientTape() as t:

        At=A[:,:,:,3:]
        Bt=B[:,:,:,3:]
        Ai=A[:,:,:,:3]
        Bi=B[:,:,:,:3]

        A=tf.concat([Ai,Bt],axis=3)

        A2B = G(A, training=True)
        A2B=tf.concat([A2B,At],axis=3)
        A2B2A = G(A2B, training=True)
        B2B = G(B, training=True)

        A2B=tf.concat([A2B[:,:,:,:3],Bt],axis=3)
        A2B_d_logits = D(A2B, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(Ai, A2B2A)
        B2B_id_loss = identity_loss_fn(Bi, B2B)

        G_loss = A2B_g_loss  + A2B2A_cycle_loss  * acycle_loss_weight   + B2B_id_loss * aidentity_loss_weight

    G_grad = t.gradient(G_loss, G.trainable_variables )
    G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables ))

    return A2B, {'A2B_g_loss': A2B_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2B_id_loss': B2B_id_loss}


@tf.function
def train_D(B, A2B):
    with tf.GradientTape() as t:
        B_d_logits = D(B, training=True)
        A2B_d_logits = D(A2B, training=True)

        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        #D_gp = gan.gradient_penalty(functools.partial(D, training=True), B, A2B, mode=agradient_penalty_mode)

        D_loss = B_d_loss + A2B_d_loss #+ D_gp * agradient_penalty_weight

    D_grad = t.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables ))

    return {'B_d_loss': B_d_loss + A2B_d_loss}


def train_step(A, B):
    A2B, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    #A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower

    D_loss_dict = train_D( B, A2B)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A,B):
    At = A[0, :, :, 3:]
    At=tf.expand_dims(At, 0)
    Bt = B[0, :, :, 3:]
    Bt=tf.expand_dims(Bt, 0)
    Ai = A[0, :, :, :3]
    Ai=tf.expand_dims(Ai, 0)

    A = tf.concat([Ai, Bt], axis=3)

    A2B = G(A, training=False)
    A2B = tf.concat([A2B, At], axis=3)
    A2B2A = G(A2B, training=False)


    return Ai,Bt,A2B[:,:,:,:3], A2B2A

def show(t,strin):
    t = tf.squeeze(t)
    image=t.numpy()
    image=(image+1)/2
    image=image*255.0
    image=image.astype('uint8')
    image2 = Image.fromarray(image, mode='RGB')
    image2.save('output\\hair\\msample\\'+strin)


# ==============================================================================
# =                                    run                                     =
# ==============================================================================
aepochs = 10

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
'''checkpoint = tl.Checkpoint(dict(G=G,
                                D=D,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)'''
# summary
# G=tf.keras.models.load_model('Models\\Generator.h5')
# D=tf.keras.models.load_model('Models\\Discriminator.h5')

train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(B_set)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(aepochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue
        count = 0

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A in tqdm.tqdm(B_set, desc='Inner Epoch Loop', total=len_dataset, position=0, leave=True):
            # for A in A_set:
            m, = A
            A = m
            ind = np.random.randint(B_length)
            B = B_list[ind]
            G_loss_dict, D_loss_dict = train_step(A, B)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations,
                       name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 500 == 0:
                G.save('Models\\Generator.h5')
                D.save('Models\\Discriminator.h5')
            if G_optimizer.iterations.numpy() % 100 == 0:
                A = next(test_iter)
                m, = A
                A = m
                ind = np.random.randint(B_length)
                B = B_list[ind]
                A, B, A2B, A2B2A = sample(A, B)
                show(A2B, 'iter_{}_{}.jpg'.format(ep, G_optimizer.iterations.numpy()))
                img = im.immerge(np.concatenate([A, A2B, A2B2A, B], axis=0), n_rows=2)
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        # checkpoint.save(ep)