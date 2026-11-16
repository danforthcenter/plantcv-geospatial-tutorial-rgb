import runpy
import shutil

import matplotlib
# The notebook sets pcv.params.debug = "plot", which makes plantcv/plantcv-geospatial show
# debug images via matplotlib. Inside a real Jupyter kernel that's harmless (an inline
# backend renders it into the notebook), but the converted script below runs as a plain
# process, where matplotlib would otherwise fall back to an interactive GUI backend (e.g.
# "macosx") and pop up a real window for every debug image, blocking the test. Force a
# non-interactive backend before anything else can import pyplot.
matplotlib.use("Agg", force=True)

import nbformat
import numpy as np
from IPython.display import display
from nbconvert import PythonExporter


class _FakeShapesLayer:
    """Stands in for a Napari Shapes layer. The notebook is unmodified for testing, so there's
    no human available to draw a polygon when it's run this way; if nothing has populated
    `.data` by the time it's read, a placeholder polygon is used instead so downstream code
    (saving to geojson, building a grid, etc.) has something valid to work with. Layers that
    *are* populated by the package itself before being read (e.g. grid lines, plot polygons)
    are unaffected, since the placeholder only kicks in while `.data` is still empty.
    """

    _PLACEHOLDER = [[0, 0], [0, 100], [100, 100], [100, 0]]

    def __init__(self):
        self._data = []
        self.shape_type = []

    def _ensure_data(self):
        if not self._data:
            self._data = [np.array(self._PLACEHOLDER)]
            self.shape_type = ["polygon"]

    @property
    def data(self):
        self._ensure_data()
        return self._data

    @data.setter
    def data(self, value):
        self._data = [np.asarray(v) for v in value]
        self.shape_type = ["polygon"] * len(self._data)

    def add(self, data, shape_type="rectangle", **kwargs):
        for d in data:
            self._data.append(np.asarray(d))
            self.shape_type.append(shape_type)


class _FakePointsLayer:
    """Stands in for a Napari Points layer. Falls back to placeholder points for the same
    reason _FakeShapesLayer does: there's no human to click points in an automated test run.
    """

    _PLACEHOLDER = [[0, 0], [0, 50], [50, 50], [50, 0], [25, 25], [10, 40]]

    def __init__(self):
        self._data = np.empty((0, 2))

    @property
    def data(self):
        if len(self._data) == 0:
            self._data = np.array(self._PLACEHOLDER, dtype=float)
        return self._data

    @data.setter
    def data(self, value):
        self._data = np.asarray(value)


class _FakeIPython:
    """Stands in for the IPython shell. nbconvert turns notebook magics such as
    `%matplotlib widget` into `get_ipython().run_line_magic(...)` calls, which fail in a
    plain Python process. Matplotlib is already forced to the Agg backend above, so the
    magics are simply ignored.
    """

    def run_line_magic(self, *args, **kwargs):
        return None

    def run_cell_magic(self, *args, **kwargs):
        return None


class _FakeViewer:
    """A minimal stand-in for napari.Viewer. Monkeypatched in for the duration of the test
    (see test_notebook below) so the notebook's InteractiveShapes/napari calls never open a
    real GUI window or need a display at all.
    """

    def __init__(self, *args, **kwargs):
        self.layers = {}

    def add_image(self, *args, **kwargs):
        return None

    def add_shapes(self, name="Shapes", **kwargs):
        layer = _FakeShapesLayer()
        self.layers[name] = layer
        return layer

    def add_points(self, name="Points", **kwargs):
        layer = _FakePointsLayer()
        self.layers[name] = layer
        return layer

    def close(self):
        pass


def test_notebook(tmpdir, monkeypatch):
    """Test the notebook by converting it to a plain Python script with nbconvert, then
    running that script with napari.Viewer mocked out, so the tutorial notebook itself never
    needs to change for testing and no real GUI/Qt window is ever opened.
    """
    import napari

    monkeypatch.setattr(napari, "Viewer", _FakeViewer)

    # Convert the notebook to an equivalent Python script
    with open("index.ipynb", "r") as f:
        nb = nbformat.read(f, as_version=4)
    source, _ = PythonExporter().from_notebook_node(nb)

    script_path = tmpdir.join("index_converted.py")
    script_path.write(source)

    # Run the converted script. Relative paths in the notebook (img/, shape_files/, output
    # files) are resolved against the current working directory, so run it from a temporary
    # copy of the repo. The notebook writes geojson/csv/json outputs (built from the
    # placeholder shapes above), and this keeps them from overwriting committed files.
    workdir = tmpdir.join("repo")
    shutil.copytree(".", str(workdir), ignore=shutil.ignore_patterns(".git"))
    monkeypatch.chdir(workdir)
    runpy.run_path(
        str(script_path),
        init_globals={"get_ipython": _FakeIPython, "display": display},
        run_name="__main__",
    )
