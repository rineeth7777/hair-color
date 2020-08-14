import multiprocessing

import tensorflow as tf


def fuse(img,path):
    s = path.numpy().decode("utf-8")
    s = s.split("\\")[-1]
    #print('fuse s \n',type(img),img)
    img2 = tf.io.read_file('data\\tests\\0.1_color\\' + s)
    img2 = tf.image.decode_png(img2, 3)
    #print('fuse 222222222 \n', type(img2), img2)
    #img2 = tf.math.scalar_mul(1/255,img2)
    img3 = tf.concat([img, img2], axis=2)
    #print('img ',img)
    return img3
def batch_data(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        #n_map_threads = multiprocessing.cpu_count()
        n_map_threads=tf.data.experimental.AUTOTUNE

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly

    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)


    dataset = dataset.prefetch(n_prefetch_batch)

    return dataset


def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        #n_map_threads = multiprocessing.cpu_count()
        n_map_threads=tf.data.experimental.AUTOTUNE
    #if shuffle and shuffle_buffer_size is None:
    shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly
    #if shuffle:
    #dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

    '''if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)'''

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    #dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset


def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None,work=0):
    """Batch dataset of memory data.

    Parameters
    ----------
    memory_data : nested structure of tensors/ndarrays/lists

    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
    if(work==0):
        dataset = batch_dataset(dataset,
                                batch_size,
                                drop_remainder=drop_remainder,
                                n_prefetch_batch=n_prefetch_batch,
                                filter_fn=filter_fn,
                                map_fn=map_fn,
                                n_map_threads=n_map_threads,
                                filter_after_map=filter_after_map,
                                shuffle=shuffle,
                                shuffle_buffer_size=shuffle_buffer_size,
                                repeat=repeat)
    else:
        dataset = batch_data(dataset,
                             batch_size,
                             drop_remainder=drop_remainder,
                             n_prefetch_batch=n_prefetch_batch,
                             filter_fn=filter_fn,
                             map_fn=map_fn,
                             n_map_threads=n_map_threads,
                             filter_after_map=filter_after_map,
                             shuffle=shuffle,
                             shuffle_buffer_size=shuffle_buffer_size,
                             repeat=repeat)

    return dataset


def disk_image_batch_dataset(img_paths,color,
                             batch_size,
                             labels=None,
                             drop_remainder=True,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None,work=0):
    """Batch dataset of disk image for PNG and JPEG.

    Parameters
    ----------
    img_paths : 1d-tensor/ndarray/list of str
    labels : nested structure of tensors/ndarrays/lists

    """
    if labels is None:
        memory_data = (img_paths,color)
    else:
        memory_data = (img_paths, labels)

    def parse_fn(path, color):
        print('pathhhhh \n',path)
        print('colorrr\n',color)
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, 3)  # fix channels to 3
        img = tf.image.resize(img, [128, 128])
        img2 = tf.io.read_file(color)
        img2 = tf.image.decode_png(img2, 3)
        img2 = tf.image.resize(img2, [128, 128])
        img3 = tf.concat([img, img2], axis=2)
        img3=tf.cast(img3,tf.float32)
        img3 = tf.clip_by_value(img3, 0, 255) / 255.0
        img3=img3*2-1

        '''s=path.decode("utf-8").split('\\')[-1]
        img2 = tf.io.read_file('data\\test\\0.1_color\\'+s)
        img2 = tf.image.decode_png(img2, 3)
        img3=tf.concat([img,img2],axis=2)'''
        return (img3,) 

    '''if map_fn:  # fuse `map_fn` and `parse_fn`
        def map_fn_(*args):
            return map_fn(*parse_fn(*args))
    else:'''
    map_fn_ = parse_fn

    dataset = memory_data_batch_dataset(memory_data,
                                        batch_size,
                                        drop_remainder=drop_remainder,
                                        n_prefetch_batch=n_prefetch_batch,
                                        filter_fn=filter_fn,
                                        map_fn=map_fn_,
                                        n_map_threads=n_map_threads,
                                        filter_after_map=filter_after_map,
                                        shuffle=shuffle,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat,work=work)

    return dataset
