class DefaultConfig(object):
    train_dataset_path = r'./datasets/train.cpkl'
    val_dataset_path = r'./datasets/val.cpkl'
    test_dataset_path = r'./datasets/test.cpkl'
    save_path = r'./models_epi/models_saved' 
    path = r'./models_saved_'

    epochs = 50
    feature_dim = 1599
    learning_rate = 0.0005
    weight_decay = 5e-4
    dropout_rate = 0.5
    batch_size = 32
    neg_wt = 0.1
    # NodeAverage
    hidden_dim = [256,512]

    # BiLSTM
    num_hidden = 32
    num_layer = 1
    # mlp
    mlp_dim = 512
