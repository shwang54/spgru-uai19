import tensorflow as tf
import matplotlib.image as mpimg
import tictoc
from data_handler_my import *
from gru_tensorflow_test_np_v3_predictor_only import GRU
from my_utilities import filecreation
import scipy.io as sio
import ConfigParser

# set random seeds
tf.set_random_seed(2016)
np.random.seed(2016)

# read model config file
model = ReadModelProto(sys.argv[1])

# ReadDataProto(sys.argv[2]) contains information about the dataset.
# Appropriately handle the dataset provided by its path
# ReadDataProto(sys.argv[2]) =
#   num_frames: 20
#   dataset_type: BOUNCING_MNIST
#   image_size: 64
#   num_digits: 2
#   step_length: 0.1
train_data = ChooseDataHandler(ReadDataProto(sys.argv[2]))
train_data_info = ReadDataProto(sys.argv[2])

# Model / Experiment related parameter configuration
exp_config = ConfigParser.ConfigParser()
exp_config.read(sys.argv[4])
exp_config_filename = sys.argv[4]

batch_size            = exp_config.getint('model parameters', 'batch_size')               # sample batch size. batch_num
num_frames            = train_data_info.num_frames                                        # sequence length. step_num x 2 since step_num is for enc which equals dec/fut
step_num              = num_frames / 2                                                    # input sequence length = num_frames / 2 = 10
image_size            = train_data_info.image_size                                        # image size. NOT the same as the input size.
elem_num              = image_size*image_size                                             # flattened input size. 64*64 = 4096
hidden_num            = exp_config.getint('model parameters', 'hidden_num')               # hidden state size
max_iters             = exp_config.getint('model parameters', 'max_iters')                # max iteration
save_images           = exp_config.getboolean('saving parameters', 'save_images')         # image saving iteration
image_save_iter       = exp_config.getint('saving parameters', 'image_save_iter')
save_checkpoints      = exp_config.getboolean('saving parameters', 'save_checkpoints')    # checkpoint saving iteration
checkpoint_save_iter  = exp_config.getint('saving parameters', 'checkpoint_save_iter')
save_matfiles         = exp_config.getboolean('saving parameters', 'save_matfiles')       # save .mat files of predictions/trains
test_iter             = exp_config.getint('model parameters', 'test_iter')                # test iteration
output_iter           = exp_config.getint('model parameters', 'output_iter')              # output iteration
valid_set_size        = exp_config.getint('model parameters', 'valid_set_size')

p_input               = tf.placeholder(tf.float32, [batch_size, step_num, elem_num])      # placeholder of each batch. batch_size x step_num x elem_num = 80 x 10 x 4096
p_inputs              = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]      # a list of input tensors with size (batch_size x elem_num)
p_input_s             = tf.placeholder(tf.float32, [batch_size, step_num, elem_num])      # placeholder of each batch. batch_size x step_num x elem_num = 80 x 10 x 4096
p_inputs_s            = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]      # a list of input tensors with size (batch_size x elem_num)
p_future              = tf.placeholder(tf.float32, [batch_size, step_num, elem_num])
p_futures             = [tf.squeeze(t, [1]) for t in tf.split(p_future, step_num, 1)]

loss_mask               = tf.placeholder(tf.float32, [step_num])

COLORBAR_MAX            = 0.7

valid_seq_input_length  = step_num*image_size*image_size
valid_seq_future_length = step_num*image_size*image_size
valid_size              = batch_size*step_num*image_size*image_size*2

# -------------------- TRAJECTORY CONFIGURATION --------------------
# Angle (degrees): 0=right, 90=down, 180=left, 270=up
# (x,y) in [0,1]: (0,0) upper left, (1,0) upper right, (0,1) bottom left, (1,1) bottom right
# delta angle: changes deflection ratio. high delta_angle -> low v_x, high v_y, vice versa
# delta_angle=0: normal deflection, delta_angle=1: only y-direction deflection, delta_angle=-1: only x-direction deflection

train_angle       = exp_config.getfloat('train trajectory', 'train_angle')
train_delta_angle = exp_config.getfloat('train trajectory', 'train_delta_angle')
train_x_low       = exp_config.getfloat('train trajectory', 'train_x_low')
train_x_high      = exp_config.getfloat('train trajectory', 'train_x_high')
train_y_low       = exp_config.getfloat('train trajectory', 'train_y_low')
train_y_high      = exp_config.getfloat('train trajectory', 'train_y_high')
train_speed       = exp_config.getfloat('train trajectory', 'train_speed')    # speed of the trajectory, not the training model

test_angle        = exp_config.getfloat('test trajectory', 'test_angle')
test_delta_angle  = exp_config.getfloat('test trajectory', 'test_delta_angle')
test_x_low        = exp_config.getfloat('test trajectory', 'test_x_low')
test_x_high       = exp_config.getfloat('test trajectory', 'test_x_high')
test_y_low        = exp_config.getfloat('test trajectory', 'test_y_low')
test_y_high       = exp_config.getfloat('test trajectory', 'test_y_high')
test_speed        = exp_config.getfloat('test trajectory', 'test_speed')

# Setup a
train_x_list      = np.random.uniform(train_x_low, train_x_high, valid_set_size+batch_size)
train_y_list      = np.random.uniform(train_y_low, train_y_high, valid_set_size+batch_size)
test_x_list       = np.random.uniform(test_x_low, test_x_high, valid_set_size+batch_size)
test_y_list       = np.random.uniform(test_y_low, test_y_high, valid_set_size+batch_size)

# reset the seed. not really important
tf.set_random_seed(2016)
np.random.seed(2016)
# ----------------------------------------- PARAMETERS END -----------------------------------------------
# --------------------------------------------------------------------------------------------------------

global_step = tf.Variable(0, trainable=False)

# My GRU
ae = GRU(hidden_num, p_inputs, p_inputs_s, p_futures, loss_mask, reverse=True, global_step=global_step, input_keep_prob=1.0, output_keep_prob=1.0)

logs_path = './log/gru-np/'
# summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

savepath = ""
if save_images:
  savepath        = filecreation('results', 'gru-np-predictor')

checkpoint_path = ""
if save_checkpoints:
  checkpoint_path = filecreation('checkpoints', 'gru-np-predictor')

saver           = tf.train.Saver(max_to_keep=2)

tictoc.tic()
with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
  sess.run(tf.global_variables_initializer())

  # --------------------------------------------------------------------------------------------------------------------
  # ----------------------------------------------- VISUALIZATION SETUP ------------------------------------------------
  # Initialize subplots so we can use set_data for repeated plotting inside the loop
  num_samples = 10
  imshow_samples = np.round((batch_size-1)*np.random.rand(num_samples,1)).astype(int)

  fig, axarr = plt.subplots(2 * num_samples, 2 * step_num, figsize=(23,15))
  fig_arr = [[mpimg._AxesImageBase for j in range(2 * step_num)] for i in range(2 * num_samples)]
  for ii in range(2 * num_samples):
    for jj in range(2 * step_num):
      rand_mat = np.random.rand(10, 10)
      axarr[ii, jj].axis('off')
      # if ii % 2 == 0 and jj >= step_num:    # of four 10-length sequences, top right shows colorbar
      if ii % 2 == 1 and jj < step_num:       # bottom left shows colorbar
      # if jj >= step_num:                    # right frames show colorbar
        fig_arr[ii][jj] = axarr[ii, jj].imshow(rand_mat, clim=(0.0, COLORBAR_MAX))
        plt.colorbar(fig_arr[ii][jj], ax=axarr[ii, jj], ticks=[])
      # elif ii % 2 == 1 and jj >= step_num:
      #   fig_arr[ii][jj] = axarr[ii, jj].imshow(rand_mat, clim=(0.0, 1.0))
      #   plt.colorbar(fig_arr[ii][jj], ax=axarr[ii, jj], ticks=[])
      else:
        fig_arr[ii][jj] = axarr[ii, jj].imshow(rand_mat, cmap='gray')
  # ----------------------------------------------- VISUALIZATION SETUP ENDS -------------------------------------------
  # --------------------------------------------------------------------------------------------------------------------


  # --------------------------------------------------------------------------------------------------------------------
  # ----------------------------------------------- MAIN ITERATION -----------------------------------------------------
  for i in range(max_iters):
    # actual training with randomly generated samples should have complete randomness without being consistent.
    # while still following the trainset configuration
    train_x_main      = np.random.uniform(train_x_low, train_x_high, batch_size)
    train_y_main      = np.random.uniform(train_y_low, train_y_high, batch_size)
    v_train_cpu, _ = train_data.GetControlledBatch(train_angle, train_x_main, train_y_main, train_delta_angle, train_speed)
    num_train, train_seq_length = v_train_cpu.shape
    train_seq_input_length = train_seq_length / 2
    train_seq_future_length = train_seq_length / 2


    v_train_input   = v_train_cpu[:,:train_seq_input_length]      # train input 1:10 frames
    v_train_future  = v_train_cpu[:,train_seq_future_length:]     # test output 11:20 frames
    v_train_input   = v_train_input.reshape([batch_size, step_num, elem_num])
    # v_train_input_s = np.zeros([batch_size, step_num, elem_num])
    # v_train_input_s[:,1:,:] = v_train_input[:,1:,:] - v_train_input[:,:step_num-1,:]
    v_train_input_s = v_train_input.var(0)
    v_train_input_s = np.tile(v_train_input_s, (batch_size, 1, 1))
    v_train_input_s = v_train_input_s * (v_train_input > 0)
    v_train_future  = v_train_future.reshape([batch_size, step_num, elem_num])

    train_loss_mask = np.ones([step_num], dtype=np.float32)
    # train_loss_mask[0] = 1

    input_list = {
      p_input: v_train_input,
      p_input_s: v_train_input_s,
      p_future: v_train_future,
      loss_mask: train_loss_mask
    }

    feed_dict = dict(input_list)

    # import pdb; pdb.set_trace()

    loss_m, loss_s, _ = sess.run([ae.loss, ae.loss_s, ae.train], feed_dict=feed_dict)
    # loss_val, _ = sess.run([ae.loss, ae.train], feed_dict={p_inputs: random_sequences, ae.p_futures: future_sequences})
    if (i+1) % output_iter == 0:
      print "iter %d:" % (i+1), "loss_s: ", loss_s, "loss_m: ", loss_m

    # --------------------------------------------------------------------------------------------------------
    # ----------------------------------------- TEST SETUP ---------------------------------------------------
    if (i+1) % test_iter == 0 or i == max_iters:

      # initialize matlab matfiles for output analysis. Test configuration first.
      matfile_test = np.zeros((valid_set_size,), dtype=np.object)

      # --------------------------------------------------------------------------------------------------------
      # ----------------------------------------- TEST TRAJECTORY SETUP -------------------------------------------
      # Setup testing with same samples every time.
      valid_idx         = 0
      valid_mat_idx     = 0       # separate index for matlab saving
      train_data.ResetTestIndex() # reset sample indices to always test on the same generated samples
      total_mean_loss   = 0
      total_var_loss    = 0
      test_batch_size   = batch_size

      # Store mean variance of test cases for each frame
      mean_variance_test = np.zeros(step_num)

      # Test iteration starts
      while valid_idx < valid_set_size:

        # The test batch is generated from training data.
        # MUST hardcopy contents
        test_x = np.array(test_x_list[valid_idx:(valid_idx+batch_size)], copy=True)
        test_y = np.array(test_y_list[valid_idx:(valid_idx+batch_size)], copy=True)
        v_valid_cpu, _ = train_data.GetControlledTestBatch(test_angle, test_x, test_y, test_delta_angle, test_speed)

        num_valid, valid_seq_length = v_valid_cpu.shape

        # test input 1:10 frames
        v_valid_input   = v_valid_cpu[:, :valid_seq_input_length]
        v_valid_input_s = v_valid_input.var(0)
        v_valid_input_s = np.tile(v_valid_input_s, (batch_size, 1, 1))
        # test target output 11:20 frames
        v_valid_future  = v_valid_cpu[:, valid_seq_future_length:]

        # only use 1:test_batch_size in case test_batch_size < batch_size
        v_valid_input   = v_valid_input.reshape([batch_size, step_num, elem_num])
        v_valid_input_s = v_valid_input_s.reshape([batch_size, step_num, elem_num])
        v_valid_input_s = v_valid_input_s * (v_valid_input > 0)
        v_valid_future  = v_valid_future.reshape([batch_size, step_num, elem_num])

        input_list = {
          p_input: v_valid_input,
          p_input_s: v_valid_input_s,
          p_future: v_valid_future}

        feed_dict = dict(input_list)

        dec_output_, dec_s_output_, loss_m, loss_s = \
          sess.run([ae.dec_output_, ae.dec_s_output_, ae.loss, ae.loss_s], feed_dict=feed_dict)

        total_mean_loss = total_mean_loss + loss_m
        total_var_loss = total_var_loss + loss_s

        # Reshape everything and only count the test_batch_size many
        input_images        = v_valid_input.reshape([batch_size, step_num, image_size, image_size])
        future_images       = v_valid_future.reshape([batch_size, step_num, image_size, image_size])
        input_images        = input_images[:test_batch_size,:,:,:]
        future_images       = future_images[:test_batch_size,:,:,:]

        dec_output_images   = dec_output_.reshape([batch_size, step_num, image_size, image_size])
        dec_s_output_images = dec_s_output_.reshape([batch_size, step_num, image_size, image_size])
        dec_output_images   = dec_output_images[:test_batch_size,:,:,:]
        dec_s_output_images = dec_s_output_images[:test_batch_size,:,:,:]

        # compute the variance of each frame. Not for visualization.
        # sum the var along image axis and samples then average by valid_set_size which computes the entire average
        mean_variance_test  = mean_variance_test + dec_s_output_images.sum(axis=2).sum(axis=2).sum(axis=0) / valid_set_size


        # --------------------------------------------------------------------------------------------------------
        # ----------------------------------------- VISUALIZATION ------------------------------------------------
        # visualize the first num_sample of the test set
        if valid_idx == 0:
          for idx in range(num_samples/2):
            for seq_j in range(step_num):
              # test encoding sequence 1:10 frames
              image_j_input = np.squeeze(input_images[idx, seq_j, :, :])
              axarr[2 * idx, seq_j].clear()
              axarr[2 * idx, seq_j].imshow(image_j_input, cmap='gray', clim=(0.0, 1.0))
              axarr[2 * idx, seq_j].axis('off')

              # test decoder output 1:10 decoded frames
              image_j_future = np.squeeze(future_images[idx, seq_j, :, :])
              axarr[2 * idx, seq_j + step_num].clear()
              axarr[2 * idx, seq_j + step_num].imshow(image_j_future, cmap='gray', clim=(0.0, 1.0))
              axarr[2 * idx, seq_j + step_num].axis('off')

              # test future sequence 11:20 frames
              image_j_dec_s_output_imscale = np.squeeze(dec_s_output_images[idx, seq_j, :, :])
              axarr[2 * idx + 1, seq_j].clear()
              axarr[2 * idx + 1, seq_j].imshow(image_j_dec_s_output_imscale)
              axarr[2 * idx + 1, seq_j].axis('off')

              # test future output 11:20 predicted frames. DECODER OUTPUT FOR NOW
              image_j_dec_output = np.squeeze(dec_output_images[idx, seq_j, :, :])
              axarr[2 * idx + 1, seq_j + step_num].clear()
              axarr[2 * idx + 1, seq_j + step_num].imshow(image_j_dec_output, cmap='gray', clim=(0.0, 1.0))
              axarr[2 * idx + 1, seq_j + step_num].axis('off')

          plt.pause(0.1)
        # -------------------------------------- VISUALIZATION ENDS ----------------------------------------------
        # --------------------------------------------------------------------------------------------------------

        valid_idx = valid_idx + test_batch_size

        if (test_batch_size + valid_idx) > valid_set_size:
          test_batch_size = valid_set_size - valid_idx


        # --------------------------------------------------------------------------------------------------------
        # ------------------------------------- TESTCONFIG MATFILE SAVE -----------------------------------------
        if save_matfiles:
          for i_mat in range(test_batch_size):
            matfile_test[valid_mat_idx] = {}
            matfile_test[valid_mat_idx]['input']        = np.squeeze(input_images[i_mat, :, :, :])          # test encoding sequence 1:10 frames
            matfile_test[valid_mat_idx]['future']       = np.squeeze(future_images[i_mat, :, :, :])         # 11:20 future frames
            matfile_test[valid_mat_idx]['fut_output']   = np.squeeze(dec_output_images[i_mat, :, :, :])     # test decoder output 1:10 decoded frames
            matfile_test[valid_mat_idx]['fut_s_output'] = np.squeeze(dec_s_output_images[i_mat, :, :, :])   # test decoder output 1:10 decoded frames

            valid_mat_idx = valid_mat_idx + 1

      if save_matfiles:
        sio.savemat(checkpoint_path + '/gru-controlled-mnist-testconfig-v1.mat',
                      {'test_config_output':matfile_test,
                       'train_angle':train_angle, 'test_angle':test_angle,
                       'train_x_main':train_x_main, 'train_y_main':train_y_main,
                       'train_x_list':train_x_list, 'train_y_list':train_y_list,
                       'test_x_list':test_x_list, 'test_y_list':test_y_list,
                       'train_speed': train_speed,
                       'test_speed': test_speed,
                       'config_filename': exp_config_filename})
      # ------------------------------------------- TEST WHILE END ----------------------------------------------
      # --------------------------------------------------------------------------------------------------------

      # --------------------------------------------------------------------------------------------------------
      # ----------------------------------------- TRAIN TRAJECTORY SETUP -------------------------------------------
      # initialize matlab matfiles for output analysis. Test configuration first.
      matfile_train = np.zeros((valid_set_size,), dtype=np.object)

      # Compute the mean variance of the test following train trajectory
      valid_idx                     = 0
      valid_mat_idx                 = 0   # separate index for matlab saving
      train_data.ResetTestIndex()
      total_mean_loss_train_config  = 0
      total_var_loss_train_config   = 0

      test_batch_size = batch_size

      # Store mean variance of test cases for each frame
      mean_variance_train_config = np.zeros(step_num)

      # Test iteration starts
      while valid_idx < valid_set_size:

        # The test batch is generated from training data.
        train_x = np.array(train_x_list[valid_idx:valid_idx+batch_size], copy=True)
        train_y = np.array(train_y_list[valid_idx:valid_idx+batch_size], copy=True)
        v_valid_cpu, _ = train_data.GetControlledTestBatch(train_angle, train_x, train_y, train_delta_angle, train_speed)

        num_valid, valid_seq_length = v_valid_cpu.shape

        # test input 1:10 frames
        v_valid_input   = v_valid_cpu[:, :valid_seq_input_length]
        v_valid_input_s = v_valid_input.var(0)
        v_valid_input_s = np.tile(v_valid_input_s, (batch_size, 1, 1))
        # test target output 11:20 frames
        v_valid_future  = v_valid_cpu[:, valid_seq_future_length:]

        v_valid_input   = v_valid_input.reshape([batch_size, step_num, elem_num])
        v_valid_input_s = v_valid_input_s.reshape([batch_size, step_num, elem_num])
        v_valid_input_s = v_valid_input_s * (v_valid_input > 0)
        v_valid_future  = v_valid_future.reshape([batch_size, step_num, elem_num])

        input_list = {
          p_input: v_valid_input,
          p_input_s: v_valid_input_s,
          p_future: v_valid_future}

        feed_dict = dict(input_list)

        dec_output_, dec_s_output_, loss_m, loss_s = \
          sess.run([ae.dec_output_, ae.dec_s_output_, ae.loss, ae.loss_s], feed_dict=feed_dict)

        total_mean_loss = total_mean_loss + loss_m
        total_var_loss = total_var_loss + loss_s

        # Reshape everything and only count the test_batch_size many
        input_images        = v_valid_input.reshape([batch_size, step_num, image_size, image_size])
        future_images       = v_valid_future.reshape([batch_size, step_num, image_size, image_size])
        input_images        = input_images[:test_batch_size,:,:,:]
        future_images       = future_images[:test_batch_size,:,:,:]

        dec_output_images   = dec_output_.reshape([batch_size, step_num, image_size, image_size])
        dec_s_output_images = dec_s_output_.reshape([batch_size, step_num, image_size, image_size])
        dec_output_images   = dec_output_images[:test_batch_size,:,:,:]
        dec_s_output_images = dec_s_output_images[:test_batch_size,:,:,:]

        # compute the variance of each frame. Not for visualization.
        # sum the var along image axis and samples then average by valid_set_size which computes the entire average
        mean_variance_train_config = mean_variance_train_config + dec_s_output_images.sum(axis=2).sum(axis=2).sum(axis=0) / valid_set_size

        # --------------------------------------------------------------------------------------------------------
        # ----------------------------------------- VISUALIZATION ------------------------------------------------
        # visualize the first num_sample of the test set
        if valid_idx == 0:
          for idx in range(num_samples/2,num_samples):
            for seq_j in range(step_num):
              # test encoding sequence 1:10 frames
              image_j_input = np.squeeze(input_images[idx-num_samples/2, seq_j, :, :])
              axarr[2 * idx, seq_j].clear()
              axarr[2 * idx, seq_j].imshow(image_j_input, cmap='gray', clim=(0.0, 1.0))
              axarr[2 * idx, seq_j].axis('off')

              # test decoder output 1:10 decoded frames
              image_j_future = np.squeeze(future_images[idx-num_samples/2, seq_j, :, :])
              axarr[2 * idx, seq_j + step_num].clear()
              axarr[2 * idx, seq_j + step_num].imshow(image_j_future, cmap='gray', clim=(0.0, 1.0))
              axarr[2 * idx, seq_j + step_num].axis('off')

              # test future sequence 11:20 frames
              image_j_dec_s_output_imscale = np.squeeze(dec_s_output_images[idx-num_samples/2, seq_j, :, :])
              axarr[2 * idx + 1, seq_j].clear()
              axarr[2 * idx + 1, seq_j].imshow(image_j_dec_s_output_imscale)
              axarr[2 * idx + 1, seq_j].axis('off')

              # test future output 11:20 predicted frames. DECODER OUTPUT FOR NOW
              image_j_dec_output = np.squeeze(dec_output_images[idx-num_samples/2, seq_j, :, :])
              axarr[2 * idx + 1, seq_j + step_num].clear()
              axarr[2 * idx + 1, seq_j + step_num].imshow(image_j_dec_output, cmap='gray', clim=(0.0, 1.0))
              axarr[2 * idx + 1, seq_j + step_num].axis('off')

          plt.pause(0.1)

          if ((i + 1) % image_save_iter == 0) & save_images:
            plt.savefig(savepath + '/iter' + str(i + 1) + '.png')
            # -------------------------------------- VISUALIZATION ENDS ----------------------------------------------
            # --------------------------------------------------------------------------------------------------------

        valid_idx = valid_idx + test_batch_size

        if (test_batch_size + valid_idx) > valid_set_size:
          test_batch_size = valid_set_size - valid_idx


        # --------------------------------------------------------------------------------------------------------
        # ------------------------------------- TRAINCONFIG MATFILE SAVE -----------------------------------------
        if save_matfiles:
          for i_mat in range(test_batch_size):
            matfile_train[valid_mat_idx] = {}
            matfile_train[valid_mat_idx]['input']         = np.squeeze(input_images[i_mat, :, :, :])        # test encoding sequence 1:10 frames
            matfile_train[valid_mat_idx]['future']        = np.squeeze(future_images[i_mat, :, :, :])       # 11:20 future frames
            matfile_train[valid_mat_idx]['fut_output']    = np.squeeze(dec_output_images[i_mat, :, :, :])   # test decoder output 1:10 decoded frames
            matfile_train[valid_mat_idx]['fut_s_output']  = np.squeeze(dec_s_output_images[i_mat, :, :, :]) # test decoder output 1:10 decoded frames

            valid_mat_idx = valid_mat_idx + 1

      if save_matfiles:
        sio.savemat(checkpoint_path + '/gru-controlled-mnist-trainconfig-v1.mat',
                      {'train_config_output':matfile_train,
                       'train_angle':train_angle, 'test_angle':test_angle,
                       'train_x_main':train_x_main, 'train_y_main':train_y_main,
                       'train_x_list':train_x_list, 'train_y_list':train_y_list,
                       'test_x_list':test_x_list, 'test_y_list':test_y_list,
                       'train_speed': train_speed,
                       'test_speed': test_speed,
                       'config_filename': exp_config_filename})

      # output numbers
      print "test result iter %d:" % (i+1), "total_mean_loss: ", total_mean_loss, "total_var_loss: ", total_var_loss
      # print "train: angle",  train_angle, ", x", train_x, ", y", train_y, ", delta_angle", train_delta_angle
      # print "test: angle",  test_angle, ", x", test_x, ", y", test_y, ", delta_angle", test_delta_angle

      mean_variance_train_config = [str(a) for a in mean_variance_train_config]
      mean_variance_test = [str(a) for a in mean_variance_test]
      print "mean_variance_train", '  '.join(mean_variance_train_config)
      print "mean_variance_test", '  '.join(mean_variance_test)
      tictoc.toc()

    # ------------------------------------------- TEST SET ENDS ----------------------------------------------
    # --------------------------------------------------------------------------------------------------------

    # save checkpoints
    if ((i+1) % checkpoint_save_iter == 0) & save_checkpoints:
        saver.save(sess, checkpoint_path + '/iter' + str(i+1) + '.ckpt')

  # ------------------------------------------- MAIN ITERATION ENDS ----------------------------------------------------
  # --------------------------------------------------------------------------------------------------------------------

import pdb; pdb.set_trace()

