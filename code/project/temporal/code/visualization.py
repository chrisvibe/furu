from project.boolean_reservoir.code.visualization import plot_train_history, plot_predictions_and_labels


def plot_many_things(model, dataset, history, evaluation='test'):
    y_test = dataset.data['y_' + f'{evaluation}'][:500]
    y_hat_test = model(dataset.data['x_' + f'{evaluation}'][:500])
    plot_train_history(model.save_path, history)
    plot_predictions_and_labels(model.save_path, y_hat_test, y_test, tolerance=model.T.accuracy_threshold, axis_limits=[0, 1])
