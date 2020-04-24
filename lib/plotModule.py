import pydotplus
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.image as img

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image  


class ScatterSlider:
    def __init__(self):
        pass

    def plot(self, df, coluna_x, coluna_y, coluna_steps, coluna_cor, colors, title):
        steps_range = np.sort(df[coluna_steps].unique())

        # Create figure
        fig = go.Figure()
        # Add traces, one for each slider step
        for step in steps_range:
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    mode = 'markers',
                    name="𝜈 = " + str(step),
                    marker={'color': list(
                        map(colors.get, df[df[coluna_steps] == step][coluna_cor]))},
                    y=df[df[coluna_steps] == step][coluna_y],
                    x=df[df[coluna_steps] == step][coluna_x]
                )
            )

        # Make 10th trace visible
        fig.data[0].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                label=str(steps_range[i]),
                method="restyle",
                args=["visible", [False] * len(fig.data)],
            )
            step["args"][1][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Intervalo: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            title=title,
            yaxis_title=coluna_y,
            xaxis_title=coluna_x
        )

        fig.show()


class ConfusionMatrix:
    def __init__(self):
        pass

    def plot(self, cm, classes, title, fig, subplot_index):    
        normalize=True
        cmap=plt.cm.Blues

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ax = fig.add_subplot(1,2,subplot_index)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax


class TreePlot:
    def __init__(self):
        pass

    def plot(self, classifier, img_name, feature_labels, class_labels, generate_img):    
        if(generate_img):
            dot_data = StringIO()
            export_graphviz(classifier, out_file=dot_data,  
                            filled=True, rounded=True,
                            special_characters=True, feature_names = feature_labels,class_names=class_labels)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
            graph.write_png(img_name)
            Image(graph.create_png())

        plt.figure().set_size_inches([16, 16])
        plt.axis('off')
        plt.imshow(img.imread(fname=img_name))
        plt.show()


class AUC:
    def __init__(self):
        pass

    def plot(self, classifier, X_train, y_train, X_test, y_test, fig, title, subplot_index):
    
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        
        y_train_auc = roc_auc_score(y_train, y_train_pred)
        y_test_auc = roc_auc_score(y_test, y_test_pred)

        y_train_proba = pd.DataFrame(classifier.predict_proba(X_train))[1].tolist()
        y_test_proba = pd.DataFrame(classifier.predict_proba(X_test))[1].tolist()

        y_train_fpr, y_train_tpr, _ = roc_curve(y_train, y_train_proba)
        y_test_fpr, y_test_tpr, _ = roc_curve(y_test, y_test_proba)

        plt.subplot(2, 2, subplot_index)
        plt.title(title)

        # Plot roc train line
        plt.plot(y_train_fpr, y_train_tpr)

        # Plot roc test line
        plt.plot(y_test_fpr, y_test_tpr)

        # Plot "no skill" line
        plt.plot([0, 1], [0, 1], linestyle='--')

        # legend
        legenda = plt.legend(['treino: %.2f' % y_train_auc, 'teste: %.2f' % y_test_auc, 'no skill: 0.50'])
        renderizador = fig.canvas.get_renderer()
        shift = max([t.get_window_extent(renderizador).width for t in legenda.get_texts()])
        for t in legenda.get_texts():
            t.set_ha('right') 
            t.set_position((shift, 0))
