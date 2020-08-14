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

py.arg('--dataset', default='summer2winter_yosemite')
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
args = py.args()

# output_dir
output_dir = py.join('output', args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)



def show(t):
    t = tf.squeeze(t)
    image=t.numpy()
    image=(image+1)/2
    image=image*255.0
    image=image.astype('uint8')
    image2 = Image.fromarray(image, mode='RGB')
    image2.show(title='real image')


def rev(img):
    img = tf.squeeze(img)
    img1 = img[:, :, :3]
    img2 = img[:, :, 3:]
    #show(img1)
    img1 = img1.numpy()
    img1 = (img1 + 1)/2
    img1 = img1 * 255.0  # or img = tl.minmax_norm(img)
    img1 = tf.convert_to_tensor(img1)
    img1=tf.dtypes.cast(img1,tf.uint8)
    img2 = img2.numpy()
    img2 = (img2 + 1)/2
    img2 = img2 * 255.0
    img2 = tf.convert_to_tensor(img2)
    img2=tf.dtypes.cast(img2, tf.uint8)
    img3 = tf.concat([img1, img2], axis=2)
    return img3


'''def prepro(img,training=0):  # preprocessing
    img=tf.squeeze(img)
    img1=img[:,:,:3]
    img2=img[:,:,3:]
    #show(img1)
    if(training==1):
        img1 = tf.image.random_flip_left_right(img1)
    img1=img1.numpy()
    img1 = img1 / 255.0  # or img = tl.minmax_norm(img)
    img1 = img1 * 2 - 1
    img1=tf.convert_to_tensor(img1)
    #show(img1)
    img2=img2.numpy()
    img2 = img2 / 255.0  # or img = tl.minmax_norm(img)
    img2 = img2 * 2 - 1
    img2=tf.convert_to_tensor(img2)
    img3=tf.concat([img1,img2],axis=2)

    return img3'''


def exchange(image,path,show=0):
    #imag = image.numpy()
    image=tf.squeeze(image)
    p1=image[:, :, :3]
    p3 = image[:, :, 3:]
    img2 = tf.io.read_file(path)
    img2 = tf.image.decode_png(img2, 3)
    img2 = tf.cast(img2, tf.float32)
    img2 = tf.clip_by_value(img2, 0, 255) / 255.0
    img2 = img2 * 2 - 1
    img3 = tf.concat([p1, img2], axis=2)

    if(show==1):
        p1t=p1.numpy()
        image2 = Image.fromarray(p1t, mode='RGB')
        image2.show(title='real image')
        p1t = img2.numpy()
        image2 = Image.fromarray(p1t, mode='RGB')
        image2.show(title='new color')
        p1t = p3.numpy()
        image2 = Image.fromarray(p1t, mode='RGB')
        image2.show(title='old color')
    return img3,p3
def createdata(l):
    n=len(l)
    lis=[]
    for k in range(n):
        s=l[k].split('\\')[-1]
        s2='data\\tests\\0.1_color\\'+s
        lis.append(s2)
    return lis


'''A_imgs = py.glob(py.join('data', '0.1_image'), '*.jpg')
A_color = py.glob(py.join('data', '0.1_color'), '*.jpg')
print(A_imgs)
print('len',len(A_imgs))'''


'''img = tf.io.read_file('C:\\Users\\M.RINEETH\\Documents\\AI\\dataset\\summer2winter_yosemite\\trainB\\2005-08-10 13_03_58.jpg')
img = tf.image.decode_png(img, 3)
img2 = tf.io.read_file('C:\\Users\\M.RINEETH\\Documents\\AI\\dataset\\summer2winter_yosemite\\trainB\\2005-09-01 10_05_22.jpg')
img2 = tf.image.decode_png(img2, 3)
img3=tf.concat([img,img2],axis=2)
print('tensor ',img3.shape)
imag=img3.numpy()
image2 = Image.fromarray(imag[:,:,:3],mode='RGB')
image2.show()
image3 = Image.fromarray(imag[:,:,2:5],mode='RGB')
image3.show()
#print('imag \n',imag)

adjhfgg'''

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A_imgs = py.glob(py.join('data\\tests', 'trainA'), '*.jpg')
len_dataset=len(A_imgs)


A_color=createdata(A_imgs)
A_Set=data.make_dataset( A_imgs,A_color,args.batch_size, args.load_size, args.crop_size, training=False,work=0)

B_imgs = py.glob(py.join('data\\tests', 'trainB'), '*.jpg')
B_color=createdata(B_imgs)
B_set=data.make_dataset( B_imgs,B_color,args.batch_size, args.load_size, args.crop_size, training=False,work=0)

B_lis=list(B_set.as_numpy_iterator())
B_list=[]
for b in B_lis:
    m,=b
    B_list.append(tf.convert_to_tensor(m))

B_length=len(B_list)
#print('b list\n',B_list[0][0].squeeze().shape)

dev = py.glob(py.join('data\\tests', 'dev'), '*.jpg')
dev_color=createdata(dev)
devset=data.make_dataset( dev,dev_color,args.batch_size, args.load_size, args.crop_size, training=False,work=0)

test_imgs = py.glob(py.join('data\\tests', 'test'), '*.jpg')
test_color=createdata(test_imgs)
test_set=data.make_dataset( test_imgs,test_color,args.batch_size, args.load_size, args.crop_size, training=False,work=0)
test_lis=list(test_set.as_numpy_iterator())
test_list=[]
for b in test_lis:
    m,=b
    test_list.append(tf.convert_to_tensor(m))


test_length=len(test_list)



#A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
#B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
#A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size, args.crop_size, training=True, repeat=False)

A2B_pool = data.ItemPool(args.pool_size)
#B2A_pool = data.ItemPool(args.pool_size)

#A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
#B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
#A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False, repeat=True)




# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 6))

D= module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 6))
#print('generator')
#print(G.summary())

#print('discriminator')
#print(D.summary())


d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)


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

        A2B=tf.concat([A2B[:,:,:,:3],Bt])
        A2B_d_logits = D(A2B, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        A2B2A_cycle_loss = cycle_loss_fn(Ai, A2B2A)
        B2B_id_loss = identity_loss_fn(Bi, B2B)

        G_loss = A2B_g_loss  + A2B2A_cycle_loss  * args.cycle_loss_weight   + B2B_id_loss * args.identity_loss_weight

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
        #D_gp = gan.gradient_penalty(functools.partial(D, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_loss = B_d_loss + A2B_d_loss #+ D_gp * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables ))

    return {'B_d_loss': B_d_loss + A2B_d_loss,
            'D_gp': D_gp}


def train_step(A, B):
    A2B, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    #A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower

    D_loss_dict = train_D( B, A2B)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A,B):
    At = A[0, :, :, 3:]
    Bt = B[0, :, :, 3:]
    Ai = A[0, :, :, :3]
    Bi = B[0, :, :, :3]

    A = tf.concat([Ai, Bt], axis=2)

    A2B = G(A, training=False)
    A2B = tf.concat([A2B, At], axis=2)
    A2B2A = G(A2B, training=False)


    return Ai,Bt,A2B, A2B2A


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G=G,
                                D=D,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(devset)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)


# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A in tqdm.tqdm(A_Set, desc='Inner Epoch Loop', total=len_dataset):
            ind = np.random.randint(B_length)
            B = B_list[ind]
            G_loss_dict, D_loss_dict = train_step(A, B)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                A = next(test_iter)
                ind = np.random.randint(test_length)
                B = test_list[ind]
                A,B,A2B, A2B2A = sample(A, B)
                img = im.immerge(np.concatenate([A, A2B, A2B2A, B], axis=0), n_rows=2)
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        checkpoint.save(ep)
