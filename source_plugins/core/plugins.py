import numpy as np
from davidia.models.messages import (
    ImageMessage,
)


from davidia.server.plugins import SourcePlugin, SourceConfigModel, SourceDataset
from nexus_file_utils import read_plot_data


class NXSImageSourceConfig(SourceConfigModel):
    path: str


class NXSImageSourcePlugin(SourcePlugin):
    name = "NXS image source"

    def __init__(
        self,
        path: str,
    ):

        self.data, raw_labels, indices, raw_axes = read_plot_data(path)

        if self.data.ndim != 3:
            raise ValueError(
                f"NXSImageSourcePlugin requires a 3D dataset, received {self.data.shape}"
            )

        # Storage-order coordinate axes:
        #
        # axis 0: polar angle
        # axis 1: angle
        # axis 2: energy

        ordered_axes: list[np.ndarray | None] = [None] * self.data.ndim
        ordered_labels: list[str] = [""] * self.data.ndim

        for axis, label, dim_list in zip(raw_axes, raw_labels, indices):
            # Only map 1D axes that correspond to a single signal dimension
            if len(dim_list) == 1:
                dim = dim_list[0]
                if 0 <= dim < self.data.ndim and ordered_axes[dim] is None:
                    ordered_axes[dim] = axis
                    ordered_labels[dim] = label

        # Fill any unmapped dimension with integer indices (0..N-1)
        for dim, size in enumerate(self.data.shape):
            if ordered_axes[dim] is None:
                ordered_axes[dim] = np.arange(size)

        self.axes = ordered_axes
        self.axes_labels = ordered_labels

        for dim, (axis, size) in enumerate(zip(self.axes, self.data.shape)):
            if axis.ndim != 1 or axis.size != size:
                raise ValueError(
                    f"Axis {dim} shape {axis.shape} does not match data dimension {size}"
                )

        self.finished = False

    def description(self) -> str:
        return "load a 3D dataset from a NeXus file"

    def get_dataset(self) -> SourceDataset:
        return SourceDataset(
            values=self.data,
            axes=self.axes,
            axis_labels=self.axes_labels,
        )

    async def has_next(self) -> bool:
        return not self.finished

    def next_data(self) -> ImageMessage | None:
        if self.finished:
            return None

        self.finished = True

        # Original shape:
        # [polar angle, angle, energy]
        #
        # Sum over energy:
        # [polar angle, angle]
        overview = self.data.mean(axis=2)

        # Heatmap values use [y, x]:
        # y = angle
        # x = polar angle
        overview = overview.T

        logger.debug(
            "NXS overview shape: %s",
            overview.shape,
        )

        return ImageMessage(
            plot_id=self.plot_id,
            im_data=HeatmapData(
                values=overview,
                domain=(
                    float(np.nanmin(overview)),
                    float(np.nanmax(overview)),
                ),
                colour_map="Inferno",
                aspect="auto",
            ),
            plot_config={
                "x_values": self.axes[0],
                "y_values": self.axes[1],
                "x_label": self.axes_labels[0],
                "y_label": self.axes_labels[1],
                "title": "ARPES overview",
            },
        )
