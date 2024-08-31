
class Trainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.losses = {}
        self.losses1 = {}
        self.losses2 = {}

    def build_graph(self):
        ###========================== DEFINE MODEL ============================###
        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.Variable(lr_init, trainable=False)

        denoise_tag = config.net_setting.denoise_model
        ngf1 = config.net_setting.ngf[0]
        SR_tag = config.net_setting.SR_model
        ngf2 = config.net_setting.ngf[1]
        Recon_tag = config.net_setting.Recon_model
        ngf3 = config.net_setting.ngf[2]
        print('[!] Denoise:%s --- ngf:%d \n[!] SR:%s --- ngf:%d\n[!] Recon:%s --- ngf:%d' % (
        denoise_tag, ngf1, SR_tag, ngf2, Recon_tag, ngf3))
        denoise_model = eval(denoise_tag)
        sr_model = eval(SR_tag)
        recon_model = eval(Recon_tag)

        # net_tag = config.net_tag
        # input_size = np.array([img_size, img_size])* n_num if 'SA' in SR_tag else np.array([img_size, img_size])
        input_size = np.array([img_size, img_size])
        SR_size = input_size * sr_factor
        Recon_size = np.multiply(SR_size, ReScale_factor)

        self.plchdr_lf = tf.placeholder('float32', [batch_size, *input_size, 1], name='t_LFP')
        self.plchdr_SynView = tf.placeholder('float32', [batch_size, *input_size, 1], name='t_SynView')
        self.plchdr_Scan_View = tf.placeholder('float32', [batch_size, *SR_size, 1], name='t_Scan_View')
        self.plchdr_Target3D = tf.placeholder('float32', [batch_size, *Recon_size, n_slices], name='t_Target3D')
        self.plchdr_wf = tf.placeholder('float32', [batch_size, *input_size, 1], name='t_wf')

        with tf.device('/gpu:{}'.format(config.TRAIN.device)):
            self.denoise_net = denoise_model(LFP=self.plchdr_lf, output_size=input_size, sr_factor=1, angRes=n_num,
                                             reuse=tf.AUTO_REUSE, channels_interp=ngf1, name=denoise_tag)

            self.SR_net = sr_model(LFP=self.denoise_net.outputs, output_size=SR_size, sr_factor=sr_factor,
                                   angRes=n_num, reuse=tf.AUTO_REUSE, name=SR_tag, channels_interp=ngf2,
                                   normalize_mode=normalize_mode, transform_layer='SAI2Macron')
            self.Recon_net = recon_model(lf_extra=self.SR_net.outputs, n_slices=n_slices, output_size=Recon_size,
                                           is_train=True, reuse=tf.AUTO_REUSE, name=Recon_tag, channels_interp=ngf3,
                                           normalize_mode=normalize_mode, transform='SAI2ViewStack',
                                           pyrimid_num=config.net_setting.Unetpyrimid_list)

        self.denoise_net.print_params(False)
        self.SR_net.print_params(False)
        self.Recon_net.print_params(False)

        denoise_vars = tl.layers.get_variables_with_name(denoise_tag, train_only=True, printable=False)
        SR_vars = tl.layers.get_variables_with_name(SR_tag, train_only=True, printable=False)
        Recon_vars = tl.layers.get_variables_with_name(Recon_tag, train_only=True, printable=False)
        # ====================
        # loss function
        # =====================
        self.loss = 0  # initial
        # self._get_losses()    # get losses
        self.denoise_loss = 0
        self.SR_loss = 0
        self.Recon_loss = 0
        self.finetune_loss = 0

        # define SR loss

        for key in denoise_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.denoise_net.outputs, reference=self.plchdr_SynView)
            self.denoise_loss = self.denoise_loss + denoise_loss[key] * temp_loss
            self.losses.update({'Denoise_' + key: denoise_loss[key] * temp_loss})
            tf.summary.scalar(key, temp_loss)

        for key in SR_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.SR_net.outputs, reference=self.plchdr_Scan_View)
            self.SR_loss = self.SR_loss + SR_loss[key] * temp_loss
            self.losses.update({'SR_' + key: SR_loss[key] * temp_loss})

            tf.summary.scalar(key, temp_loss)

        for key in Recon_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.Recon_net.outputs, reference=self.plchdr_Target3D)
            self.Recon_loss = self.Recon_loss + Recon_loss[key] * temp_loss
            self.losses.update({'Recon_' + key: Recon_loss[key] * temp_loss})
            tf.summary.scalar(key, temp_loss)

        for key in finetune_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.Recon_net.outputs, reference=self.plchdr_wf, projection_range=config.Loss.projection_range)
            self.finetune_loss = self.finetune_loss + finetune_loss[key] * temp_loss
            self.losses1.update({'finetune_' + key: finetune_loss[key] * temp_loss})
            tf.summary.scalar(key, temp_loss)

        # self.loss_stage1 = loss_ratio[0] * self.denoise_loss + loss_ratio[1] * self.SR_loss
        # self.loss_stage2 = loss_ratio[0] * self.SR_loss + loss_ratio[1] * self.Recon_loss
        self.loss_stage1 = self.denoise_loss
        self.loss_stage2 = loss_ratio[0] * self.denoise_loss + loss_ratio[1] * self.SR_loss
        self.loss_stage3 = loss_ratio[0] * self.denoise_loss + loss_ratio[1] * self.SR_loss + loss_ratio[
            2] * self.Recon_loss
        self.loss_finetune = loss_ratio[3]*self.finetune_loss
        # self.loss = 0.1*self.denoise_loss + 0.3*self.SR_loss + 0.6*self.Recon_loss
        tf.summary.scalar('learning_rate', self.learning_rate)
        # define test_loss when test
        # self.loss_test = loss_ratio[0] * self.denoise_loss + loss_ratio[1] * self.SR_loss + loss_ratio[2] * self.Recon_loss+loss_ratio[3]*self.finetune_loss
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
        self.finetune_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss_finetune,
                                                                                               var_list=denoise_vars + SR_vars + Recon_vars)

        self.merge_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        self.fetches = self.losses
        self.fetches['opti_fuse_stage1'] = self.fuse1_optim
        self.fetches['opti_fuse_stage2'] = self.fuse2_optim
        self.fetches['opti_fuse_stage3'] = self.fuse3_optim

        self.fetches2 = self.losses1
        self.fetches2['opti_fuse_stage4'] = self.finetune_optim

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
        tl.files.exists_or_mkdir(test_lf_dir)
        tl.files.exists_or_mkdir(test_hr_dir)
        tl.files.exists_or_mkdir(test_mr_dir)
        tl.files.exists_or_mkdir(test_stack_dir)
        tl.files.exists_or_mkdir(test_lfexp_dir)
        tl.files.exists_or_mkdir(test_wf_dir)
        save_configs(save_folder=checkpoint_dir,cg=config)
        # initialize vars
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.assign(self.learning_rate, lr_init))

        if loading_pretrain_model:
            SR_ckpt_file = [filename for filename in os.listdir(pretrain_ckpt_dir) if
                            ('.npz' in filename and 'best' in filename and 'SR' in filename)]
            denoise_ckpt_file = [filename for filename in os.listdir(pretrain_ckpt_dir) if
                                 ('.npz' in filename and 'best' in filename and 'denoise' in filename)]
            recon_ckpt_file = [filename for filename in os.listdir(pretrain_ckpt_dir) if
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
            SR_ckpt = self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir, epoch=begin_epoch, label='SR')
            Recon_ckpt = self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir, epoch=begin_epoch, label='recon')
            assert SR_ckpt != None and Recon_ckpt != None, 'No ckpt has been found'
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(checkpoint_dir, denoise_ckpt),
                                         network=self.denoise_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(checkpoint_dir, SR_ckpt),
                                         network=self.SR_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(checkpoint_dir, Recon_ckpt),
                                         network=self.Recon_net)

        ###====================== LOAD DATA ===========================###
        self.training_Target3D, self.training_ScanView, self.training_SynView, self.training_lf2d, self.training_lfexp, self.training_wf=self.dataset._load_dataset()

        self._get_test_data()
        self.test_img_num=4
        self.training_pair_num = len(self.training_Target3D)
        self.wf_num = len(self.training_wf)
        self.iter_num = self.training_pair_num + self.wf_num-2*self.test_img_num
        for epoch in range(n_epoch):
            if epoch != 0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                self.sess.run(tf.assign(self.learning_rate, lr_init * new_lr_decay))
                print('\nlearning rate updated : %f\n' % (lr_init * new_lr_decay))
            for iter in range(self.iter_num):
                step_time = time.time()
                if iter % 2 == 0:
                    i = random.randint(self.test_img_num, self.training_pair_num)

                    feed_train = {self.plchdr_Target3D: self.training_Target3D[i:i+1],
                                  self.plchdr_Scan_View: self.training_ScanView[i:i+1],
                                  self.plchdr_SynView: self.training_SynView[i:i+1],
                                  self.plchdr_lf: self.training_lf2d[i:i+1]
                                  }
                    evaluated = self.sess.run(self.fetches, feed_train)
                else:
                    i = random.randint(self.test_img_num, self.wf_num)
                    feed_train2 = {self.plchdr_wf: self.training_wf[i:i+1],
                                   self.plchdr_lf: self.training_lfexp[i:i+1]
                                   }
                    evaluated = self.sess.run(self.fetches2, feed_train2)


                # learning rate update

                # log
                loss_str = [name + ':' + str(value) for name, value in evaluated.items() if 'loss' in name]
                print("\rEpoch:[%d/%d] iter:[%d/%d] time: %4.3fs ---%s" % (
                    epoch, n_epoch + begin_epoch, iter, self.iter_num, time.time() - step_time, loss_str),
                      end='')

            ##record and save checkpoints
            if epoch%10 == 0:
                tag = str(epoch)

                den_file_name = checkpoint_dir + '/denoise_net_{}.npz'.format(tag)
                sr_file_name = checkpoint_dir + '/SR_net_{}.npz'.format(tag)
                recon_file_name = checkpoint_dir + '/recon_net_{}.npz'.format(tag)

                tl.files.save_npz(self.denoise_net.all_params, name=den_file_name, sess=self.sess)
                tl.files.save_npz(self.SR_net.all_params, name=sr_file_name, sess=self.sess)
                tl.files.save_npz(self.Recon_net.all_params, name=recon_file_name, sess=self.sess)

                for i in range(test_num):
                    test_lr_batch = self.test_LFP[i:i+1]
                    test_lfexp_batch = self.test_LFexp[i:i+1]
                    denoise_view = self.sess.run(self.denoise_net.outputs, {self.plchdr_lf: test_lr_batch})
                    SR_view = self.sess.run(self.SR_net.outputs, {self.plchdr_lf: test_lr_batch})
                    Recon_stack = self.sess.run(self.Recon_net.outputs, {self.plchdr_lf: test_lr_batch})
                    recon_exp = self.sess.run(self.Recon_net.outputs, {self.plchdr_lf: test_lfexp_batch})
                    write3d(denoise_view, test_saving_dir + ('denoise_{}_%d.tif' % (i)).format(tag))
                    write3d(SR_view, test_saving_dir + ('SR_{}_%d.tif' % (i)).format(tag))
                    write3d(Recon_stack, test_saving_dir + ('Recon_{}_%d.tif' % (i)).format(tag))
                    write3d(recon_exp, test_saving_dir + ('exp_Recon_{}_%d.tif'% (i)).format(tag))

    def _get_test_data(self):
        self.test_target3d, self.test_Scan_View, self.test_Synview, self.test_LFP, self.test_LFexp, self.test_wf = self.dataset.for_test()
        for i in range(test_num):
            write3d(self.test_target3d[i:i + 1], test_stack_dir + '/Target3d_%d.tif' % i)
            write3d(self.test_Scan_View[i:i + 1], test_hr_dir + '/Scan_View_%d.tif' % i)
            write3d(self.test_Synview[i:i + 1], test_mr_dir + '/SynView_%d.tif' % i)
            write3d(self.test_LFP[i:i + 1], test_lf_dir + '/LFP_%d.tif' % i)
            write3d(self.test_LFexp[i:i + 1], test_lfexp_dir + '/LFexp_%d.tif' % i)
            write3d(self.test_wf[i:i + 1], test_wf_dir + '/wf_%d.tif' % i)

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
    import os
    from skimage import io
    from config_finetune import config
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.net_setting.gpu_idx)
    import time
    import tensorflow as tf
    import tensorlayer as tl
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import random
    from model import *
    from misc.dataset_finetune import Dataset
    from misc.utils import write3d, save_configs, is_number
    from config_finetune import config

    ###=================img pre ===========================###
    img_size = config.img_setting.img_size
    n_num = config.img_setting.Nnum
    sr_factor = config.img_setting.sr_factor
    n_slices = config.img_setting.n_slices
    ReScale_factor = config.img_setting.ReScale_factor

    base_size = img_size // n_num  # lateral size of lf_extra
    normalize_mode = config.preprocess.normalize_mode
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
    label = config.new_model_name
    ckpt_saving_interval = config.TRAIN.ckpt_saving_interval
    root_path = config.root_path
    checkpoint_dir = os.path.join(root_path, 'DL', config.TRAIN.ckpt_dir)
    log_dir = os.path.join(root_path, 'DL', config.TRAIN.log_dir)
    test_saving_dir = os.path.join(root_path, 'DL', config.TRAIN.test_saving_path)
    plot_test_loss_dir = os.path.join(test_saving_dir, 'test_loss_plt')
    test_stack_dir = os.path.join(test_saving_dir, 'Target3D')
    test_hr_dir = os.path.join(test_saving_dir, 'Scan_View')
    test_mr_dir = os.path.join(test_saving_dir, 'Clean_View')
    test_lf_dir = os.path.join(test_saving_dir, 'BG_View')
    test_lfexp_dir = os.path.join(test_saving_dir, 'exp')
    test_wf_dir = os.path.join(test_saving_dir, 'wf')
    ###=================losses define ===========================###
    denoise_loss = config.Loss.denoise_loss
    SR_loss = config.Loss.SR_loss
    Recon_loss = config.Loss.Recon_loss
    finetune_loss = config.Loss.finetune_loss
    loss_ratio = config.Loss.Ratio

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=int, default=0, help='')
    args = parser.parse_args()

    is_hdf5=config.img_setting.save_hdf5

    training_dataset = Dataset(base_path=config.img_setting.data_root_path,
                               is_hdf5=is_hdf5,
                               LFexp_path=os.path.join(config.img_setting.fine_tune_data_path,'LF'),
                               wf_path=os.path.join(config.img_setting.fine_tune_data_path,'WF'),
                               n_num=n_num,
                               shuffle_for_epoch=shuffle_for_epoch,
                               normalize_mode=normalize_mode,
                               sample_ratio=sample_ratio,
                               shuffle_all_data=config.TRAIN.shuffle_all_data)
    trainer = Trainer(training_dataset)
    trainer.build_graph()
    trainer.train(begin_epoch=args.ckpt)
