import teachableMatchines
import tensorflow

def test_1():
    model = tensorflow.keras.models.load_model('keras_model.h5')
    y_pred = teachableMatchines.predict_one(model, "Test_0")
    print("predict probability :", y_pred)
    y_pred_id = y_pred.argmax()
    assert y_pred_id, 2