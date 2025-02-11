import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from model import *
from dataset import Dataset
from config import config

###=================img pre ===========================###
img_size = config.img_setting.img_size
n_channels = config.img_setting.n_channels
sr_factor_list = config.img_setting.sr_factor
sample_ratio = config.TRAIN.sample_ratio
test_num = 4

###=================training para ===========================###
loading_pretrain_model = config.Pretrain.loading_pretrain_model
pretrain_ckpt_dir = config.Pretrain.ckpt_dir

batch_size = config.TRAIN.batch_size
shuffle_for_epoch = config.TRAIN.shuffle_for_epoch
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

###=================dir ===========================###
label = config.label
test_saving_dir = config.TRAIN.test_saving_path
checkpoint_dir = config.TRAIN.ckpt_dir
ckpt_saving_interval = config.TRAIN.ckpt_saving_interval
log_dir = config.TRAIN.log_dir
plot_test_loss_dir = os.path.join(test_saving_dir, 'test_loss_plt')

test_hr_dir = os.path.join(test_saving_dir, 'HRx%d'%sr_factor_list[2])
test_mr_dir = os.path.join(test_saving_dir, 'MRx%d'%sr_factor_list[1])
test_lr_dir = os.path.join(test_saving_dir, 'LRx%d'%sr_factor_list[0])
test_llr_dir = os.path.join(test_saving_dir, 'LLRx1')

###=================losses define ===========================###
denoise_loss = config.Loss.denoise_loss
SR_loss = config.Loss.SR_loss
Recon_loss = config.Loss.Recon_loss

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class Trainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.losses = {}

    def build_graph(self):
        ###========================== DEFINE MODEL ============================###
        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.Variable(lr_init, trainable=False)


        print('[!] Building graph...')
        base_size = np.array([img_size, img_size])
        self.plchdr_LLR = tf.placeholder('float32', [batch_size, *base_size, n_channels], name='t_LLR')
        self.plchdr_LR = tf.placeholder('float32', [batch_size, *(base_size*sr_factor_list[0]), n_channels], name='t_LR')
        self.plchdr_MR = tf.placeholder('float32', [batch_size, *(base_size*sr_factor_list[0]*sr_factor_list[1]), n_channels], name='t_MR')
        self.plchdr_HR = tf.placeholder('float32', [batch_size, *(base_size*sr_factor_list[0]*sr_factor_list[1]*sr_factor_list[2]), n_channels], name='t_HR')

        tag_list = ['DenoiseNet','SRNet','ReconNet']
        with tf.device('/gpu:{}'.format(config.TRAIN.device)):
            self.denoise_net = RCAN(lr=self.plchdr_LLR, sr_factor=sr_factor_list[0], format_out=True, reuse=False, name=tag_list[0])
            self.SR_net = RCAN(lr=self.denoise_net.outputs, sr_factor=sr_factor_list[1], format_out=True, reuse=False, name=tag_list[1])
            self.Recon_net = RCAN(lr=self.SR_net.outputs, sr_factor=sr_factor_list[2], format_out=True, reuse=False, name=tag_list[2])

        self.denoise_net.print_params(False)
        self.SR_net.print_params(False)
        self.Recon_net.print_params(False)

        denoise_vars = tl.layers.get_variables_with_name(tag_list[0], train_only=True, printable=False)
        SR_vars = tl.layers.get_variables_with_name(tag_list[1], train_only=True, printable=False)
        Recon_vars = tl.layers.get_variables_with_name(tag_list[2], train_only=True, printable=False)
        # ====================
        # loss function
        # =====================
        self.loss = 0  # initial
        self.denoise_loss = 0
        self.SR_loss = 0
        self.Recon_loss = 0

        # define SR loss

        for key in denoise_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.denoise_net.outputs, reference=self.plchdr_LR)
            self.denoise_loss = self.denoise_loss + denoise_loss[key] * temp_loss
            self.losses.update({'Denoise_' + key: denoise_loss[key] * temp_loss})
            tf.summary.scalar(key, temp_loss)

        for key in SR_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.SR_net.outputs, reference=self.plchdr_MR)
            self.SR_loss = self.SR_loss + SR_loss[key] * temp_loss
            self.losses.update({'SR_' + key: SR_loss[key] * temp_loss})
            tf.summary.scalar(key, temp_loss)

        for key in Recon_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.Recon_net.outputs, reference=self.plchdr_HR)
            self.Recon_loss = self.Recon_loss + Recon_loss[key] * temp_loss
            self.losses.update({'Recon_' + key: Recon_loss[key] * temp_loss})
            tf.summary.scalar(key, temp_loss)


        self.loss_stage1 = self.denoise_loss
        self.loss_stage2 = self.SR_loss + self.denoise_loss
        self.loss_stage3 = self.Recon_loss + self.SR_loss + self.denoise_loss
        tf.summary.scalar('learning_rate', self.learning_rate)
        # define test_loss when test
        self.loss_test = self.Recon_loss + self.SR_loss + self.denoise_loss
        # ----------------create sess-------------
        configProto = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        configProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=configProto)

        self.fuse1_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss_stage1,
                                                                                            var_list=denoise_vars)
        self.fuse2_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss_stage2,
                                                                                            var_list=denoise_vars + SR_vars)
        self.fuse3_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss_stage3,
                                                                                            var_list=denoise_vars + SR_vars + Recon_vars)

        self.merge_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

    def _train(self, begin_epoch):
        """Train the VCD-Net
        Params
            -begin_epoch: int, if not 0, a checkpoint file will be loaded and the training will continue from there
        """
        ## create folders to save result images and trained model
        save_dir = test_saving_dir
        tl.files.exists_or_mkdir(save_dir)
        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(log_dir)
        tl.files.exists_or_mkdir(plot_test_loss_dir)
        tl.files.exists_or_mkdir(test_llr_dir)
        tl.files.exists_or_mkdir(test_mr_dir)
        tl.files.exists_or_mkdir(test_lr_dir)
        tl.files.exists_or_mkdir(test_hr_dir)

        # initialize vars
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.assign(self.learning_rate, lr_init))

        if loading_pretrain_model:
            denoise_ckpt_file = [filename for filename in os.listdir(pretrain_ckpt_dir) if
                                 ('.npz' in filename and 'best' in filename and 'denoise' in filename)]

            SR_ckpt_file = [filename for filename in os.listdir(pretrain_ckpt_dir) if
                            ('.npz' in filename and 'best' in filename and 'sr' in filename)]

            recon_ckpt_file= [filename for filename in os.listdir(pretrain_ckpt_dir) if
                                 ('.npz' in filename and 'best' in filename and 'recon' in filename)]

            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(pretrain_ckpt_dir, SR_ckpt_file[0]),
                                         network=self.SR_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(pretrain_ckpt_dir, denoise_ckpt_file[0]),
                                         network=self.denoise_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(pretrain_ckpt_dir, recon_ckpt_file[0]),
                                         network=self.Recon_net)
        if (begin_epoch != 0):
            denoise_ckpt = self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir, epoch=begin_epoch,
                                                         label='denoise')
            SR_ckpt = self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir, epoch=begin_epoch, label='sr')
            Recon_ckpt = self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir, epoch=begin_epoch, label='recon')

            assert SR_ckpt != None and Recon_ckpt != None, 'No ckpt has been found'
            tl.files.load_and_assign_npz(sess=self.sess, name=denoise_ckpt, network=self.denoise_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=SR_ckpt, network=self.SR_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=Recon_ckpt, network=self.Recon_net)

        ###====================== LOAD DATA ===========================###
        dataset_size = self.dataset.prepare(batch_size, n_epoch)
        final_cursor = (dataset_size // batch_size - 1) * batch_size
        self._get_test_data()

        fetches = self.losses

        fetches['opti_fuse_stage1'] = self.fuse1_optim
        fetches['opti_fuse_stage2'] = self.fuse2_optim
        fetches['opti_fuse_stage3'] = self.fuse3_optim
        fetches['batch_summary'] = self.merge_op

        while self.dataset.hasNext():

            HR_batch, MR_batch, LR_batch, LLR_batch, cursor, epoch = self.dataset.iter()  # get data
            feed_train = {
                self.plchdr_HR: HR_batch,
                self.plchdr_MR: MR_batch,
                self.plchdr_LR: LR_batch,
                self.plchdr_LLR: LLR_batch,
            }

            epoch += begin_epoch
            step_time = time.time()

            # learning rate update
            if epoch != 0 and (epoch % decay_every == 0) and cursor == 0:
                new_lr_decay = lr_decay ** (epoch // decay_every)
                self.sess.run(tf.assign(self.learning_rate, lr_init * new_lr_decay))
                print('\nlearning rate updated : %f\n' % (lr_init * new_lr_decay))

            # infer loss
            evaluated = self.sess.run(fetches, feed_train)

            # log
            loss_str = [name + ':' + str(value) for name, value in evaluated.items() if 'loss' in name]
            print("\rEpoch:[%d/%d] iter:[%d/%d] time: %4.3fs ---%s" % (
                epoch, n_epoch + begin_epoch, cursor, dataset_size, time.time() - step_time, loss_str), end='')
            self.summary_writer.add_summary(evaluated['batch_summary'],
                                            epoch * (dataset_size // batch_size - 1) + cursor / batch_size)
            ##record and save checkpoints
            if cursor == final_cursor:
                self._record_avg_test_loss(epoch, self.sess)
                if epoch != 0 and (epoch % ckpt_saving_interval == 0):
                    self._save_intermediate_ckpt(epoch, self.sess)

    def _get_test_data(self):
        self.test_HR, self.test_MR, self.test_LR, self.test_LLR = self.dataset.for_test()
        for i in range(test_num):
            tifffile.imwrite(test_hr_dir + '/hr_%d.tif' % i,self.test_HR[i])
            tifffile.imwrite(test_mr_dir + '/mr_%d.tif' % i,self.test_MR[i])
            tifffile.imwrite(test_lr_dir + '/lr_%d.tif' % i,self.test_LR[i])
            tifffile.imwrite(test_llr_dir + '/llr_%d.tif' % i,self.test_LLR[i])

    def _save_intermediate_ckpt(self, tag, sess):
        tag = ('epoch%d' % tag) if is_number(tag) else tag

        den_file_name = checkpoint_dir + '/denoise_net_{}.npz'.format(tag)
        sr_file_name = checkpoint_dir + '/sr_net_{}.npz'.format(tag)
        recon_file_name = checkpoint_dir + '/recon_net_{}.npz'.format(tag)

        tl.files.save_npz(self.denoise_net.all_params, name=den_file_name, sess=sess)
        tl.files.save_npz(self.SR_net.all_params, name=sr_file_name, sess=sess)
        tl.files.save_npz(self.Recon_net.all_params, name=recon_file_name, sess=sess)

        if 'epoch' in tag:
            if batch_size >= test_num:
                test_lr_batch = self.test_LLR[0:batch_size]
                denoise_img = self.sess.run(self.denoise_net.outputs, {self.plchdr_LLR: test_lr_batch})
                SR_img  = self.sess.run(self.SR_net.outputs, {self.plchdr_LLR: test_lr_batch})
                Recon_img  = self.sess.run(self.Recon_net.outputs, {self.plchdr_LLR: test_lr_batch})
                for i in range(test_num):
                    tifffile.imwrite(test_saving_dir + ('denoise_{}_%d.tif' % (i)).format(tag),denoise_img[i])
                    tifffile.imwrite(test_saving_dir + ('SR_{}_%d.tif' % (i)).format(tag), SR_img[i])
                    tifffile.imwrite(test_saving_dir + ('Recon_{}_%d.tif' % (i)).format(tag),Recon_img[i])
            else:
                for idx in range(0, test_num, batch_size):
                    if idx + batch_size <= test_num:
                        test_lr_batch = self.test_LLR[idx:idx + batch_size]
                        [denoise_img, SR_img, Recon_img] = self.sess.run([self.denoise_net.outputs,self.SR_net.outputs,self.Recon_net.outputs], {self.plchdr_LLR: test_lr_batch})
                        for i in range(len(SR_img)):
                            tifffile.imwrite(
                                    test_saving_dir + ('denoise_{}_%d.tif' % (i + idx * batch_size)).format(tag),denoise_img[i:i + 1])
                            tifffile.imwrite(
                                    test_saving_dir + ('SR_{}_%d.tif' % (i + idx * batch_size)).format(tag),SR_img[i:i + 1])
                            tifffile.imwrite(
                                    test_saving_dir + ('Recon_{}_%d.tif' % (i + idx * batch_size)).format(tag),Recon_img[i:i + 1])

    def _record_avg_test_loss(self, epoch, sess):
        if 'min_test_loss' not in dir(self):
            self.min_test_loss = 1e10
            self.best_epoch = 0
            self.test_loss_plt = []

        test_loss = 0
        test_data_num = len(self.test_LLR)
        print("")
        for idx in range(0, test_data_num, batch_size):
            if idx + batch_size <= test_data_num:
                test_llr_batch = self.test_LLR[idx: idx + batch_size]
                test_LR_batch = self.test_LR[idx: idx + batch_size]
                test_MR_batch = self.test_MR[idx: idx + batch_size]
                test_HR_batch = self.test_HR[idx: idx + batch_size]

                feed_test = {self.plchdr_LLR: test_llr_batch,
                             self.plchdr_LR: test_LR_batch,
                             self.plchdr_MR: test_MR_batch,
                             self.plchdr_HR: test_HR_batch
                             }

                test_loss_batch, losses_batch = sess.run([self.loss_test, self.losses], feed_test)
                loss_str = [name + ':' + str(value) for name, value in losses_batch.items() if 'loss' in name]
                test_loss += test_loss_batch
                print('\rvalidation  [% 2d/% 2d] loss = %.6f --%s ' % (idx, test_data_num, test_loss_batch, loss_str),
                      end='')
        test_loss /= (len(self.test_LLR) // batch_size)
        print('\navg = %.6f best = %.6f (@epoch%d)' % (test_loss, self.min_test_loss, self.best_epoch))
        self.test_loss_plt.append([epoch, test_loss])
        temp_file_name = plot_test_loss_dir + '/plot_test_loss.npy'
        np.save(temp_file_name, self.test_loss_plt)

        if (test_loss < self.min_test_loss):
            self.min_test_loss = test_loss
            self.best_epoch = epoch
            self._save_intermediate_ckpt(tag='best', sess=sess)

    def _plot_test_loss(self):
        loss = np.asarray(self.test_loss_plt)
        plt.figure()
        plt.plot(loss[:, 0], loss[:, 1])
        plt.savefig(plot_test_loss_dir + '/test_loss.png', bbox_inches='tight')
        plt.show()

    def _traversal_through_ckpts(self, checkpoint_dir, epoch, label=None):
        ckpt_found = False
        filelist = os.listdir(checkpoint_dir)
        for file in filelist:
            if '.npz' in file and str(epoch) in file:
                if label is not None:
                    if label in file:
                        return file
                else:
                    return file
        return None

    def train(self, **kwargs):
        try:
            self._train(**kwargs)
        finally:
            self._plot_test_loss()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--ckpt', type=int, default=0, help='')
    parser.add_argument('-idx', '--model_idx', type=int, default='0', help='')
    parser.add_argument('--trans', type=bool, default=False, help='')
    args = parser.parse_args()
    if args.trans == False:
        training_dataset = Dataset(config.img_setting.HR+'/',
                                   config.img_setting.MR+'/',
                                   config.img_setting.LR+'/',
                                   config.img_setting.LLR+'/',
                                   shuffle_for_epoch=shuffle_for_epoch,
                                   sample_ratio=sample_ratio,
                                   shuffle_all_data=config.TRAIN.shuffle_all_data)
        trainer = Trainer(training_dataset)
        trainer.build_graph()
        trainer.train(begin_epoch=args.ckpt)
