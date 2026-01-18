import sys
import logging
import joblib
import tensorflow as tf
import pandas
import numpy as np
import matplotlib.pyplot as plt


### Script Setup ###
SCALER_PATH = 'source/models/scaler_data.bin'
MODEL_PATH = 'source/models/bearing-sensor-anomaly-detection.h5'
COMPOSED_INTERACTIONS_DATASET_PATH = 'source/data/dept-composed-dt/composed-logger-interactions.csv'
CONVEYOR_INTERACTIONS_DATASET_PATH = 'source/data/conveyor-dt/conveyor-logger-interactions.csv'
CONVEYOR_PROPERTIES_DATASET_PATH = 'source/data/conveyor-dt/conveyor-logger-properties.csv'
CONVEYOR_VIBRATIONS_DATASET_PATH = 'source/data/conveyor-dt/conveyor-logger-vibrations.csv'
SAVING_PATH = 'output/graph/'
SAVE_LAST_GENERATED_IMAGE = False
SHOW_GRAPH = False

pandas.options.mode.copy_on_write = True

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",)
logger = logging.getLogger('FT-Conveyor-Cognitive-DT-Model')


if __name__ == '__main__':
    ### Preliminary AI-model preparation ###
    logger.info('\n\n### Preliminary AI-model preparation ###')
    # Loading the scaler
    scaler = joblib.load(SCALER_PATH)
    scaler.clip = False
    model = tf.keras.models.load_model(MODEL_PATH)

    ### DT datasets retrieval and preparation ###
    logger.info('\n\n### DT datasets retrieval and manipulation ###')
    # Reading Conveyor Vibrations dataset
    conveyor_vibrations = pandas.read_csv(CONVEYOR_VIBRATIONS_DATASET_PATH, index_col=[0])
    conveyor_vibrations.index = pandas.to_datetime(conveyor_vibrations.index, unit='ms')
    conveyor_vibrations[['b1', 'b2', 'b3', 'b4']] = (
        conveyor_vibrations['value'].apply(lambda x: pandas.Series(eval(x))))
    conveyor_vibrations[['b1', 'b2', 'b3', 'b4']] = (
        conveyor_vibrations[['b1', 'b2', 'b3', 'b4']].apply(pandas.to_numeric))
    conveyor_vibrations = conveyor_vibrations.drop('value', axis=1)

    logger.info('\nConveyor vibrations dataset loaded:\n%s', conveyor_vibrations)

    # Scaling the Conveyor Vibrations dataset and generating datasets for prediction application and
    # Loss MAE plotting
    conveyor_vibrations_scaled = scaler.transform(conveyor_vibrations)
    conveyor_vibrations_for_prediction = conveyor_vibrations_scaled.reshape(
        conveyor_vibrations.shape[0],
        1,
        conveyor_vibrations.shape[1]
    )
    conveyor_vibrations_for_loss_mae = conveyor_vibrations_for_prediction.reshape(
        conveyor_vibrations_for_prediction.shape[0],
        conveyor_vibrations_for_prediction.shape[2]
    )

    # Performing the prediction
    bearing_health_prediction = model.predict(conveyor_vibrations_for_prediction)

    # Adjusting the prediction dataset
    bearing_health_prediction = bearing_health_prediction.reshape(
        bearing_health_prediction.shape[0],
        bearing_health_prediction.shape[2]
    )
    bearing_health_prediction = pandas.DataFrame(
        bearing_health_prediction, columns=['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
    )

    logger.info('\nBearing health prediction:\n%s', bearing_health_prediction)

    # Setting the dataset containing the Loss MAE calculation for each predicted point
    loss_calc_dataset = np.mean(
        np.abs(bearing_health_prediction - conveyor_vibrations_for_loss_mae),
        axis=1
    ).to_frame()
    loss_calc_dataset.rename(columns={0: 'Loss_mae'}, inplace=True)
    loss_calc_dataset['Threshold'] = 0.275      # Setting the prediction threshold

    # Merging into main dataset
    conveyor_vibrations = conveyor_vibrations.reset_index()
    conveyor_vibrations['row_number'] = conveyor_vibrations.index
    loss_calc_dataset['row_number'] = loss_calc_dataset.index
    merged_dataset = pandas.merge(
        conveyor_vibrations,
        loss_calc_dataset,
        on="row_number",
        how="outer"
    )

    # Dropping unnecessary columns
    merged_dataset.drop(columns=['b1', 'b2', 'b3', 'b4', 'row_number'], inplace=True)

    # Extrapolating columns for Loss MAE dataset graph plotting
    loss_graph_dataset = merged_dataset[['timestamp', 'Loss_mae', 'Threshold']]
    resulting_fig, resulting_ax = plt.subplots(1, 1)

    resulting_ax.plot(loss_graph_dataset['timestamp'], loss_graph_dataset['Loss_mae'], color="blue")
    resulting_ax.plot(loss_graph_dataset['timestamp'], loss_graph_dataset['Threshold'], color="red")

    # Reading Composed Interactions csv and performing dataset manipulation
    composed_interactions = pandas.read_csv(COMPOSED_INTERACTIONS_DATASET_PATH)
    composed_interactions['timestamp'] = pandas.to_datetime(
        composed_interactions['timestamp'],
        unit='ms'
    )

    logger.info(
        '\nInteractions dataset of the Department Composed DT loaded:\n%s',
        composed_interactions
    )

    # Merging into main dataset
    merged_dataset = pandas.merge(
        merged_dataset,
        composed_interactions,
        on='timestamp',
        how='outer'
    )

    # Manipulating main dataset
    merged_dataset.replace('Medium', 'true', inplace=True)
    merged_dataset['comp-requests'] = merged_dataset['payload']
    merged_dataset.drop(columns=['payload'], inplace=True)
    merged_dataset.drop(columns=['type'], inplace=True)
    merged_dataset.drop(columns=['class'], inplace=True)

    # Reading Conveyor Interactions csv and performing dataset manipulation for sub-dataset
    # extraction
    conveyor_interactions = pandas.read_csv(CONVEYOR_INTERACTIONS_DATASET_PATH, index_col=[0])
    conveyor_interactions['timestamp'] = pandas.to_datetime(
        conveyor_interactions['timestamp'],
        unit='ms'
    )
    conveyor_actions = conveyor_interactions[conveyor_interactions['class'] == 'conveyor-action']
    conveyor_piece_at_exit = (
        conveyor_interactions)[conveyor_interactions['class'] == 'conveyor-piece-at-exit']
    logger.info('\nInteractions dataset of the Conveyor DT loaded\n')
    logger.info('\nConveyor events:\n%s', conveyor_piece_at_exit)
    logger.info('\nConveyor events type:\n%s', type(conveyor_piece_at_exit))
    logger.info('\nConveyor action requests:\n%s', conveyor_actions)

    conv_to_merge = pandas.concat([conveyor_actions, conveyor_piece_at_exit])

    conv_to_merge['conv-received-ev'] = conv_to_merge['payload']
    conv_to_merge['conv-sent-act'] = conv_to_merge['payload']

    conv_to_merge_ev = conv_to_merge[conv_to_merge['type'] == 'EVENT']
    conv_to_merge_act = conv_to_merge[conv_to_merge['type'] == 'ACTION']

    conv_to_merge_ev.drop(columns=['payload', 'type', 'class', 'conv-sent-act'], inplace=True)
    conv_to_merge_act.drop(columns=['payload', 'type', 'class', 'conv-received-ev'], inplace=True)

    conv_to_merge_ev.replace('Medium', 'true', inplace=True)
    conv_to_merge_act.replace('Medium', 'true', inplace=True)

    # Dropping last line because it does not fit the graph storytelling
    conv_to_merge_act.drop(conv_to_merge_act.index[-1], inplace=True)

    # Merging into main dataset extrapolated datasets
    merged_dataset = pandas.merge(merged_dataset, conv_to_merge_ev, on='timestamp', how='outer')
    merged_dataset = pandas.merge(merged_dataset, conv_to_merge_act, on='timestamp', how='outer')

    # Reading Conveyor Health dataset and performing manipulation
    conveyor_health = pandas.read_csv(CONVEYOR_PROPERTIES_DATASET_PATH, index_col=[0])
    conveyor_health.index = pandas.to_datetime(conveyor_health.index, unit='ms')
    conveyor_health['conv-healthy'] = conveyor_health['value']
    conveyor_health.drop(columns=['value'], inplace=True)
    conveyor_health.drop(columns=['property'], inplace=True)

    logger.info('\nProperties of the Conveyor DT loaded')


    # Merging into main dataset and performing dataset manipulation
    merged_dataset = pandas.merge(merged_dataset, conveyor_health, on='timestamp', how='outer')
    merged_dataset = merged_dataset.sort_values(by='timestamp')
    merged_dataset['conv-healthy'].convert_dtypes()
    merged_dataset['conv-sent-act'].convert_dtypes()
    merged_dataset['conv-healthy'].fillna(method='ffill', inplace=True)

    # Dropping initial part of the dataset to improve plotting result
    merged_dataset = merged_dataset.drop(merged_dataset.index[:284])
    merged_dataset['Loss_mae'] = merged_dataset['Loss_mae'].fillna(method='ffill')
    merged_dataset['Threshold'] = merged_dataset['Threshold'].fillna(method='ffill')

    # Modifying timestamps format for plotting
    merged_dataset['timestamp'] = merged_dataset['timestamp'].dt.strftime('%H:%M:%S')
    merged_dataset['comp-requests'] = merged_dataset['comp-requests'].fillna('false')
    merged_dataset['conv-received-ev'] = merged_dataset['conv-received-ev'].fillna('false')
    merged_dataset['conv-sent-act'] = merged_dataset['conv-sent-act'].fillna('false')
    merged_dataset['conv-healthy'] = merged_dataset['conv-healthy'].ffill()
    merged_dataset['conv-healthy'] = merged_dataset['conv-healthy'].bfill()

    ### Plots ###
    logger.info('\n\n### Preparing plots ###')

    # Setting the final plot
    final_plot_fig, final_plot_axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    # Plotting vibrations loss mae
    final_plot_axs[0].plot(
        merged_dataset['timestamp'],
        merged_dataset['Loss_mae'],
        color="blue",
        label='conv loss mae'
    )
    final_plot_axs[0].plot(merged_dataset['timestamp'], merged_dataset['Threshold'], color="red")
    final_plot_axs[0].set_yscale('log')
    final_plot_axs[0].set_ylim([1e-2, 2e0])
    final_plot_axs[0].set_yticks([1e-2, 1e2])
    final_plot_axs[0].set_ylabel('Conveyor DT\nloss mae')
    final_plot_axs[0].set_facecolor('white')
    final_plot_axs[0].grid(color='#EAEAF2')
    final_plot_axs[0].yaxis.grid(color='white')
    final_plot_axs[0].spines['top'].set_color('black')
    final_plot_axs[0].spines['bottom'].set_color('black')
    final_plot_axs[0].spines['left'].set_color('black')
    final_plot_axs[0].spines['right'].set_color('black')

    # Plotting conv health state
    final_plot_axs[1].plot(
        merged_dataset['timestamp'],
        merged_dataset['conv-healthy'],
        color="green",
        label='conv healthy state'
    )
    final_plot_axs[1].set_yticks([0., 1.])
    final_plot_axs[1].set_yticklabels(['false', 'true'])
    final_plot_axs[1].set_ylabel('Conveyor DT\nhealthy state')
    final_plot_axs[1].set_facecolor('white')
    final_plot_axs[1].grid(color='#EAEAF2')
    final_plot_axs[1].yaxis.grid(color='white')
    final_plot_axs[1].spines['top'].set_color('black')
    final_plot_axs[1].spines['bottom'].set_color('black')
    final_plot_axs[1].spines['left'].set_color('black')
    final_plot_axs[1].spines['right'].set_color('black')
    text_to_annotate = \
        f'1) The AI inside the DT\n updates the DT state\nin accordance with the\nclassification'
    final_plot_axs[1].annotate(
        text_to_annotate,
        xy=(290, 0.97),
        xytext=(190, 0.3),
        arrowprops=dict(color='black', lw=2, arrowstyle='->', relpos=(0.9, 0.7))
    )

    # Plotting comp sent action
    marker_comp_req, stem_comp_req, baseline_comp_req = final_plot_axs[2].stem(
        merged_dataset['timestamp'],
        merged_dataset['comp-requests'],
        basefmt='b-',
        linefmt='r-',
        markerfmt='D',
        label='comp sent requests'
    )
    marker_comp_req.set_markerfacecolor('#DA63A1')
    marker_comp_req.set_markeredgecolor((1, 0, 0, 0))
    stem_comp_req.set_color('#DA63A1')
    baseline_comp_req.set_color('#DA63A1')
    final_plot_axs[2].set_ylabel('Composed DT\nsent requests')
    final_plot_axs[2].set_facecolor('white')
    final_plot_axs[2].grid(color='#EAEAF2')
    final_plot_axs[2].yaxis.grid(color='white')
    final_plot_axs[2].spines['top'].set_color('black')
    final_plot_axs[2].spines['bottom'].set_color('black')
    final_plot_axs[2].spines['left'].set_color('black')
    final_plot_axs[2].spines['right'].set_color('black')
    text_to_annotate_1 = f'2) The cDT sends\na slowdown request to\nthe conveyor DT'
    text_to_annotate_2 = f'6) The cDT\nsends the\ndepartment\nshutdown\nsignal'
    final_plot_axs[2].annotate(
        text_to_annotate_1,
        xy=(290, 0.97),
        xytext=(190, 0.3),
        arrowprops=dict(color='black', lw=2, arrowstyle='->', relpos=(0.9, 0.7))
    )
    final_plot_axs[2].annotate(
        text_to_annotate_2,
        xy=(340, 0.97),
        xytext=(350, 0.15),
        arrowprops=dict(color='black', lw=2, arrowstyle='->', relpos=(0.9, 0.5))
    )

    # Plotting conv rec ev
    marker_conv_ev, stem_conv_ev, baseline_conv_ev = final_plot_axs[3].stem(
        merged_dataset['timestamp'],
        merged_dataset['conv-received-ev'],
        basefmt='b-',
        linefmt='r-',
        markerfmt='D',
        label='conv received requests'
    )
    marker_conv_ev.set_markerfacecolor('orange')
    marker_conv_ev.set_markeredgecolor((1, 0, 0, 0))
    stem_conv_ev.set_color('orange')
    baseline_conv_ev.set_color('orange')
    final_plot_axs[3].set_ylabel('Conveyor DT\nreceived events')
    final_plot_axs[3].set_facecolor('white')
    final_plot_axs[3].grid(color='#EAEAF2')
    final_plot_axs[3].yaxis.grid(color='white')
    final_plot_axs[3].spines['top'].set_color('black')
    final_plot_axs[3].spines['bottom'].set_color('black')
    final_plot_axs[3].spines['left'].set_color('black')
    final_plot_axs[3].spines['right'].set_color('black')
    text_to_annotate_1 = f'3) The slowdown request\nis received by the\nconveyor DT'
    text_to_annotate_2 = f'5) Notification\nthat the last\npiece has\nreached the\nconveyor end'
    final_plot_axs[3].annotate(
        text_to_annotate_1,
        xy=(290, 0.97),
        xytext=(190, 0.3),
        arrowprops=dict(color='black', lw=2, arrowstyle='->', relpos=(0.9, 0.7))
    )
    final_plot_axs[3].annotate(
        text_to_annotate_2,
        xy=(340, 0.97),
        xytext=(350, 0.15),
        arrowprops=dict(color='black', lw=2, arrowstyle='->', relpos=(0.9, 0.5))
    )

    # Plotting conv sent act
    marker_conv_act, stem_conv_act, baseline_conv_act = final_plot_axs[4].stem(
        merged_dataset['timestamp'],
        merged_dataset['conv-sent-act'],
        basefmt='b-',
        linefmt='r-',
        markerfmt='D',
        label='conv sent actions'
    )
    marker_conv_act.set_markerfacecolor('#FFC813')
    marker_conv_act.set_markeredgecolor((1, 0, 0, 0))
    stem_conv_act.set_color('#FFC813')
    baseline_conv_act.set_color('#FFC813')
    final_plot_axs[4].set_ylabel('Conveyor DT\nsent actions')
    final_plot_axs[4].set_facecolor('white')
    final_plot_axs[4].grid(color='#EAEAF2')
    final_plot_axs[4].yaxis.grid(color='white')
    final_plot_axs[4].spines['top'].set_color('black')
    final_plot_axs[4].spines['bottom'].set_color('black')
    final_plot_axs[4].spines['left'].set_color('black')
    final_plot_axs[4].spines['right'].set_color('black')
    text_to_annotate = \
        f'4) The slowdown action\nis sent from conveyor DT\nto its physical counterpart'
    final_plot_axs[4].annotate(
        text_to_annotate,
        xy=(290, 0.97),
        xytext=(190, 0.3),
        arrowprops=dict(color='black', lw=2, arrowstyle='->', relpos=(0.9, 0.7))
    )

    # Adjusting x-ticks
    current_ticks = final_plot_axs[4].get_xticklabels()
    selected_ticks = []

    for pos, tick in enumerate(current_ticks):
      if pos % 25 == 0:
        selected_ticks.append(tick.get_text())

    final_plot_axs[4].set_xticks(selected_ticks)

    for tick in final_plot_axs[4].get_xticklabels():
        tick.set_rotation(90)

    # Adjusting plotting layout
    # final_plot_fig.tight_layout(pad=20)

    # Save the plot
    if SAVE_LAST_GENERATED_IMAGE:
        logging.info('Saving the breakdown strategy timeline .png')
        final_plot_fig.savefig(SAVING_PATH + 'breakdown-strategy-timeline.png')
        logging.info('Plot saved')
    # Show the plot
    if SHOW_GRAPH:
        logging.info('Showing the image')
        final_plot_fig.show()

    logger.info('\n\n### Graph generation completed ###')
