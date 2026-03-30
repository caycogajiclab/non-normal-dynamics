import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def plot_ground_truth(sample,T,D,rotated_W):

    # Create subplot layout
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[
            "Sample Initial Gaussian Pulse",
            "Sample Outputs (separate axes)",
            "Sample Outputs (all on same axis)",
            "",
	    "Schur Form T",
            "Rotation Matrix D",
            "M_in = D @ T @ D.T",
            "Eigenvalues of T (red) and W (blue)"
        ]
    )


    # ---- Panel 1: Initial Gaussian Pulse ----
    fig.add_trace(
        go.Scatter(y=np.asarray(sample[0]).ravel(), mode="lines"),
        row=1, col=1
    )

    # ---- Panel 2: Outputs (separate axes) ----
    for n in range(sample[1].shape[1]):
        fig.add_trace(
            go.Scatter(
                y=sample[1][:, n] - n,
                mode="lines",
                showlegend=False
            ),
            row=1, col=2
        )

    # Remove y-axis ticks for panel 2
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    # ---- Panel 3: Outputs (all on same axis) ----
    for n in range(sample[1].shape[1]):
        fig.add_trace(
            go.Scatter(
                y=sample[1][:, n],
                mode="lines",
                showlegend=False
            ),
            row=1, col=3
        )

    # ---- Panel 4: Schur heatmap ----

    fig.add_trace(
        go.Heatmap(
            z=T,
        colorscale=[
    [0, "#354ad2"],
    [0.5, "#fff3ed"],
    [1, "#d82c09"]
    ]),
        row=2, col=1
    )

    # ---- Panel 5: Rotation Matrix heatmap ----

    fig.add_trace(
        go.Heatmap(
            z=D,
        colorscale=[
    [0, "#354ad2"],
    [0.5, "#fff6e8"],
    [1, "#d82c09"]
    ]),
        row=2, col=2
    )

    # ---- Panel 6: rotated Schur ----

    fig.add_trace(
        go.Heatmap(
            z=rotated_W,
        colorscale=[
    [0, "#354ad2"],
    [0.5, "#fff6e8"],
    [1, "#d82c09"]
    ]),
        row=2, col=3
    )

    fig.update_yaxes(autorange="reversed",row=2)

    # ---- Panel 7: Eigenvalues ----

    ew = np.linalg.eigvals(rotated_W)
    et = np.linalg.eigvals(T)

    fig.add_trace(
        go.Scatter(
            x=ew.real,
            y=ew.imag,
            mode="markers",
            name="eig(W)",
            marker=dict(color="blue", size=8)
        ),
        row=2, col=4
    )

    # eig(T)
    fig.add_trace(
        go.Scatter(
            x=et.real,
            y=et.imag,
            mode="markers",
            name="eig(T)",
            marker=dict(color="red", size=8, opacity=0.5)
        ),
        row=2, col=4
    )

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 400)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            name="unit circle",
            line=dict(color="black", dash="dash", width=1)
        ),
        row=2, col=4
    )

    # Axis formatting for eigenvalue plot
    fig.update_xaxes(
        title_text="Real",
        scaleanchor="y",
        scaleratio=1,
        row=2, col=4
    )
    fig.update_yaxes(
        title_text="Imag",
        row=2, col=4
    )





    fig.update_layout(
        width=1000,
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig




def plot_interm_fig(W,output):

        fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Learned Weights",
            "Sample Prediction"
        ]
    )

        fig.add_trace(
            go.Heatmap(
                z=W,
            colorscale=[
        [0, "#354ad2"],
        [0.5, "#fff3ed"],
        [1, "#d82c09"]
        ]),
            row=1, col=1
        )

        for n in range(output.shape[1]):
            fig.add_trace(
                go.Scatter(
                    y=output[:, n],
                    mode="lines",
                    showlegend=False
                ),
                row=1, col=2
            )

        fig.update_yaxes(autorange="reversed",row=1,col=1)

        return fig
