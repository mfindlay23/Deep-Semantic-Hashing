import os
import scipy
import dotmap
import tqdm

def getSession(gpu_num, gpu_fraction):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

    num_threads = os.environ.get("OMP_NUM_THREADS")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)



