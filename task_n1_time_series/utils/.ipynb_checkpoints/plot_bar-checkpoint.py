import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


def plot_ts_pred(og_ts, pred_ts, model_name=None, og_ts_opacity = 0.5, pred_ts_opacity = 0.5):
    """
    Plot plotly time series of the original (og_ts) and predicted (pred_ts) time series values to check how our model performs.
    model_name: name of the model used for predictions
    og_ts_opacity: opacity of the original time series
    pred_ts_opacity: opacity of the predicted time series
    """
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = og_ts.index, y = np.array(og_ts.values), name = "Observed",
                         line_color = 'deepskyblue', opacity = og_ts_opacity))

    try:
        fig.add_trace(go.Scatter(x = pred_ts.index, y = pred_ts, name = model_name,
                         line_color = 'lightslategrey', opacity = pred_ts_opacity))
    except: #if predicted values are a numpy array they won't have an index
        fig.add_trace(go.Scatter(x = og_ts.index, y = pred_ts, name = model_name,
                         line_color = 'lightslategrey', opacity = pred_ts_opacity))


    #fig.add_trace(go)
    fig.update_layout(title_text = 'Observed test set vs predicted energy MWH values using {}'.format(model_name),
                  xaxis_rangeslider_visible = True)
    fig.show()


def plot_hist_loss(hst):
    hst.test_hist_loss
    plt.figure(figsize=(15, 7.5), layout='constrained')
    plt.plot(list(range(len(hst.test_hist_loss))), hst.test_hist_loss, label='test_loss')  # etc.
    plt.plot(list(range(len(hst.train_hist_loss))), hst.train_hist_loss, label='train_loss')  # etc.
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Train process")
    plt.legend()