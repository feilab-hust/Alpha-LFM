
class Trainer:
    def __init__(self,dataset):
        self.losses = {}
        self.dataset=dataset
    def build_graph(self):
        ###========================== DEFINE MODEL ============================###
        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.Variable(lr_init, trainable=False)

        input_size = np.array([img_size, img_size])
        SR_size = input_size * sr_factor
        Recon_size = np.multiply(SR_size, ReScale_factor)

        Recon_tag = local_configs.net_setting.Recon_model
        ngf3 = local_configs.net_setting.ngf[2]
        recon_model=eval(Recon_tag)

        self.plchdr_lf = tf.placeholder('float32', [batch_size, *SR_size, 1],name='plh_lfm')
        self.plchdr_Target3D = tf.placeholder('float32', [batch_size, *Recon_size, n_slices],name='plh_Target3D')

        with tf.device('/gpu:{}'.format(local_configs.TRAIN.device)):
            self.Recon_net = recon_model(lf_extra=self.plchdr_lf, n_slices=n_slices, output_size=Recon_size,
                                         is_train=True, reuse=False, name=Recon_tag, channels_interp=ngf3,
                                         normalize_mode=local_configs.preprocess.normalize_mode, transform='SAI2ViewStack',
                                         pyrimid_list=local_configs.net_setting.Unetpyrimid_list)
        self.Recon_net.print_params(False)
        g_vars = tl.layers.get_variables_with_name(Recon_tag, train_only=True, printable=False)

        # ====================
        # loss function
        # =====================
        self.loss = 0  # initial

        for key in loss_dict:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.Recon_net.outputs, reference=self.plchdr_Target3D)
            self.loss = self.loss + loss_dict[key] * temp_loss
            self.losses.update({'Recon_' + key: loss_dict[key] * temp_loss})

        # define test_loss when test
        self.loss_test = self.loss

        # ----------------create sess-------------
        configProto = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        configProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=configProto)
        self.optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss, var_list=g_vars)


    def _train(self, begin_epoch):
        """Train the VCD-Net
        Params
            -begin_epoch: int, if not 0, a checkpoint file will be loaded and the training will continue from there
        """
        save_dir = test_saving_dir
        tl.files.exists_or_mkdir(save_dir)
        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(test_lf_dir)
        tl.files.exists_or_mkdir(test_hr_dir)

        #initialize vars
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.assign(self.learning_rate, lr_init))


        ###====================== LOAD DATA ===========================###

        dataset_size = self.dataset.prepare(batch_size, n_epoch)
        final_cursor = (dataset_size // batch_size - 1) * batch_size
        self._get_test_data()

        fetches = self.losses
        fetches['optim'] = self.optim

        while self.dataset.hasNext():
            # step_time = time.time()
            HR_batch, LF_batch, cursor, epoch = self.dataset.iter()  # get data
            # print('fetch_time:%.4f'%(time.time() - step_time))
            feed_train = {self.plchdr_lf: LF_batch, self.plchdr_Target3D: HR_batch}
            epoch += begin_epoch
            step_time = time.time()

            # learning rate update
            if epoch != 0 and (epoch % decay_every == 0) and cursor == 0:
                new_lr_decay = lr_decay ** (epoch // decay_every)
                self.sess.run(tf.assign(self.learning_rate, lr_init * new_lr_decay))
                print('\nlearning rate updated : %f\n' % (lr_init * new_lr_decay))
            # infer
            evaluated = self.sess.run(fetches, feed_train)
            # log
            loss_str = [name + ':' + str(value) for name, value in evaluated.items() if 'loss' in name]
            print("\rEpoch:[%d/%d] iter:[%d/%d] time: %4.3fs ---%s" % (
                epoch, n_epoch+begin_epoch, cursor, dataset_size, time.time() - step_time, loss_str), end='')

            ##record and save checkpoints
            if cursor == final_cursor:
                self._record_avg_test_loss(epoch, self.sess)
                if epoch != 0 and (epoch % ckpt_saving_interval == 0):
                    self._save_intermediate_ckpt(epoch, self.sess)

    def _get_test_data(self):
        self.test_target3d, self.test_lf_extra = self.dataset.for_test()
        for i in range(test_num):
            write3d(self.test_target3d[i:i + 1], test_hr_dir + '/target3d%d.tif' % i)
            write3d(self.test_lf_extra[i:i + 1], test_lf_dir + '/lf_extra%d.tif' % i)


    def _save_intermediate_ckpt(self, tag, sess):
        tag = ('epoch%d' % tag) if is_number(tag) else tag
        npz_file_name = checkpoint_dir + '/recon_net_{}.npz'.format(tag)
        tl.files.save_npz(self.Recon_net.all_params, name=npz_file_name, sess=sess)
        if 'epoch' in tag:
            if batch_size >= test_num:
                test_lr_batch = self.test_lf_extra[0:batch_size]
                out = self.sess.run(self.Recon_net.outputs, {self.plchdr_lf: test_lr_batch})
                for i in range(test_num):
                    write3d(out[i:i + 1], test_saving_dir + ('/test_{}_%d.tif' % (i)).format(tag))
            else:
                for idx in range(0, test_num, batch_size):
                    if idx + batch_size <= test_num:
                        test_lr_batch = self.test_lf_extra[idx:idx + batch_size]
                        out = self.sess.run(self.Recon_net.outputs, {self.plchdr_lf: test_lr_batch})
                        for i in range(len(out)):
                            write3d(out[i:i + 1],
                                    test_saving_dir + ('/test_{}_%d.tif' % (i + idx * batch_size)).format(tag))
    def _record_avg_test_loss(self, epoch, sess):
        if 'min_test_loss' not in dir(self):
            self.min_test_loss = 1e10
            self.best_epoch = 0
            self.test_loss_plt = []

        test_loss = 0
        test_data_num = len(self.test_lf_extra)
        print("")


        for idx in range(0, test_data_num, batch_size):
            if idx + batch_size <= test_data_num:
                test_lf_batch = self.test_lf_extra[idx: idx + batch_size]
                test_hr_batch = self.test_target3d[idx: idx + batch_size]
                feed_test = {self.plchdr_lf: test_lf_batch, self.plchdr_Target3D: test_hr_batch}


                test_loss_batch, losses_batch = sess.run([self.loss_test, self.losses], feed_test)
                loss_str = [name + ':' + str(value) for name, value in losses_batch.items() if 'loss' in name]
                test_loss += test_loss_batch

                print('\rvalidation  [% 2d/% 2d] loss = %.6f --%s ' % (idx, test_data_num, test_loss_batch, loss_str),
                      end='')


        test_loss /= (len(self.test_lf_extra) // batch_size)
        print('\navg = %.6f best = %.6f (@epoch%d)' % (test_loss, self.min_test_loss, self.best_epoch))

        if (test_loss < self.min_test_loss):
            self.min_test_loss = test_loss
            self.best_epoch = epoch
            self._save_intermediate_ckpt(tag='best', sess=sess)
            # self._save_pb(sess)

    def _plot_test_loss(self):
        pass
        # loss = np.asarray(self.test_loss_plt)
        # plt.figure()
        # plt.plot(loss[:, 0], loss[:, 1])
        # plt.savefig(plot_test_loss_dir + '/test_loss.png', bbox_inches='tight')
        # plt.show()

    def train(self, **kwargs):
        try:
            self._train(**kwargs)
        finally:
            self._plot_test_loss()

    def _find_available_ckpt(self, end):
        begin = end
        while not os.path.exists(checkpoint_dir + '/recon_epoch{}.npz'.format(begin)):
            begin -= 10
            if begin < 0:
                return 0
        print('\n\ninit ckpt found at epoch %d\n\n' % begin)
        tl.files.load_and_assign_npz(sess=self.sess, name=checkpoint_dir + '/denoise_epoch{}.npz'.format(begin),
                                     network=self.Recon_net)
        return begin
if __name__ == '__main__':
        import argparse
        import os
        import sys
        parser = argparse.ArgumentParser()
        parser.add_argument('-g', '--gpu', type=int, default=0, help='')
        parser.add_argument('-cfg', '--config_path', type=str)
        args = parser.parse_args()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print('current folder', os.getcwd())
        sys.path.append(os.path.join(os.getcwd(),'DL'))
        import time
        import tensorflow as tf
        import tensorlayer as tl
        import numpy as np
        from model import *
        from misc.utils import write3d,_raise,is_number
        from misc.dataset_LF_Vol import Dataset
        from config import configs_settings

        ###=================img pre ===========================###
        local_configs = configs_settings(args.config_path)
        img_size = local_configs.img_setting.img_size
        n_slices = local_configs.img_setting.n_slices
        n_num = local_configs.img_setting.Nnum
        ReScale_factor = local_configs.img_setting.ReScale_factor
        base_size = img_size // n_num  # lateral size of lf_extra
        test_num = 2
        sample_ratio = 1.0
        sr_factor = local_configs.img_setting.sr_factor
        ###=================training para ===========================###

        loss_dict = {
            'mse_loss': 1.0,
            'edge_loss': 0.1,
        }
        batch_size = 1
        shuffle_for_epoch = 1
        lr_init = 5 * 1e-4
        beta1 = 0.9
        n_epoch = local_configs.Pretrain.Training_epoch[2]
        lr_decay = 0.5
        decay_every = 50
        ###=================dir ===========================###
        root_path = local_configs.root_path
        label = local_configs.label + '_preVCD'
        test_saving_dir = os.path.join(root_path, 'DL', local_configs.TRAIN.test_saving_path, 'preVCD')
        checkpoint_dir = os.path.join(root_path, 'DL', local_configs.TRAIN.ckpt_dir, 'preVCD')
        ckpt_saving_interval = 10
        test_hr_dir = os.path.join(test_saving_dir, 'Syn_view')
        test_lf_dir = os.path.join(test_saving_dir, 'LF2D_TEST')
        save_hdf5 = local_configs.img_setting.save_hdf5
        data_root_path = local_configs.img_setting.data_root_path
        to_disk=local_configs.TRAIN.to_Disk
        finish_flag_file = os.path.join(root_path,'logging','preVCD_finish_%s.txt'%local_configs.label)
        if os.path.exists(finish_flag_file):
            os.remove(finish_flag_file)

        if save_hdf5:
            base_path = os.path.join(local_configs.img_setting.data_root_path, 'training_data.h5')
            data_str = ['sVol', 'sLF']
        else:
            base_path = os.path.join(local_configs.img_setting.data_root_path)
            data_str = ['sVol', 'sLF']
        training_dataset = Dataset(base_path=base_path,
                                   data_str=data_str,
                                   n_slices=n_slices,
                                   n_num=n_num,
                                   lf2d_base_size=base_size,
                                   shuffle_for_epoch=shuffle_for_epoch,
                                   normalize_mode=local_configs.preprocess.normalize_mode, sample_ratio=sample_ratio, save_hdf5=save_hdf5)
        trainer = Trainer(training_dataset)
        trainer.build_graph()
        trainer.train(begin_epoch=0)
        print('vcd:',args.gpu)
        with open(finish_flag_file, 'w') as f:
            f.write('1')




