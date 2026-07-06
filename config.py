class DefaultConfig(object):
    train_dataset_path = r'./datasets/train.cpkl'
    val_dataset_path = r'./datasets/val.cpkl'
    test_dataset_path = r'./datasets/test.cpkl'
    save_path = r'./models_epi/models_saved_test' 
    path = r'./models_saved_'

    epochs = 50
    seeds = [42, 3407, 0, 1234, 2026]
    use_k_fold = True
    k_folds = 5
    feature_dim = 1343
    learning_rate = 0.0005
    weight_decay = 5e-4
    use_early_stop = False
    early_stop_patience = 10
    dropout_rate = 0.5
    batch_size = 32
    neg_wt = 0.1
    # NEAGE
    hidden_dim = [256,512]

    # Lightweight sequence encoder
    num_hidden = 32
    num_layer = 1
    sequence_encoder_dim = 64
    use_bilstm = True
    transformer_layers = 2
    use_transformer = True
    use_hybrid_pooling = True
    # mlp
    mlp_dim = 512

    # Interaction switches:
    #   use_cross_attention refines sequence/global and graph/local features
    #   before prediction.
    #   use_contrastive adds a cross-modal contrastive loss. When both are
    #   enabled, contrastive learning is applied after cross attention.
    use_cross_attention = True
    use_contrastive = True

    # Contrastive learning
    #   'in_batch' keeps the original batch-wise negatives.
    #   'cross_protein' uses only residues from other proteins/graphs as negatives.
    #   'cluster' clusters residues in the shared space; same-cluster samples are positives
    #   and different-cluster samples are negatives.
    contrastive_negative_mode = 'cluster'
    contrastive_memory_size = 4096
    contrastive_num_prototypes = 8
    contrastive_prototype_iters = 5
    contrastive_dim = 128
    contrastive_temperature = 0.1
    contrastive_initial_weight = 0.5
    contrastive_decay_rate = 5.0
    label_aware_contrastive = True
