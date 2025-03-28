from .cnn import CNN_Predictor
from .fcn import FCN_Predictor
from .lstm import LSTM_Predictor, StackedLSTM_Predictor, CNNLSTM_Predictor
from .gru import GRU_Predictor, StackedGRU_Predictor, CNNGRU_Predictor
from .jit_gru import JitGRU_Predictor, CNNJitGRU_Predictor, JitSeq2Seq_Predictor
from .tcn import TCN_Predictor
from .grad_to_input import (
    GradToInputNN,
    ImprovedGradToInputNN,
    GradToInputNN_Sigmoid,
    GradToTemporalInputNN,
    ImprovedGradToInputNN_2,
    ImprovedGradToInputNN_3,
    ImprovedGradToInputNN_Probabilistic,
    ImprovedGradToInputNN_Quantile,
)


def initialize_model_by_name(model_name, device="cpu", **kwargs):
    """
    Initialize a model by its name.

    :param model_name: The name of the model to initialize.
    :param kwargs: Additional arguments to pass to the model's constructor.
    :return: An instance of the model.
    """
    if model_name in model_classes:
        model = model_classes[model_name](**kwargs)
        model.to(device)
        return model
    else:
        raise ValueError(f"Model '{model_name}' not found.")


model_classes = {
    LSTM_Predictor.name: LSTM_Predictor,
    StackedLSTM_Predictor.name: StackedLSTM_Predictor,
    CNNLSTM_Predictor.name: CNNLSTM_Predictor,
    GRU_Predictor.name: GRU_Predictor,
    StackedGRU_Predictor.name: StackedGRU_Predictor,
    CNNGRU_Predictor.name: CNNGRU_Predictor,
    JitGRU_Predictor.name: JitGRU_Predictor,
    CNNJitGRU_Predictor.name: CNNJitGRU_Predictor,
    CNN_Predictor.name: CNN_Predictor,
    FCN_Predictor.name: FCN_Predictor,
    TCN_Predictor.name: TCN_Predictor,
}


def get_fc_state_dict(model_state_dict, fc_variable_name="fc."):
    ordered_dict = {}
    for key, value in model_state_dict.items():
        if fc_variable_name in key:
            ordered_dict[key.replace(fc_variable_name, "")] = value
    return ordered_dict
