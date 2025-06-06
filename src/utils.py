import nbformat

# Cargar ambos notebooks
with open("src/explore_sergio.ipynb") as f:
    nb_c = nbformat.read(f, as_version=4)

with open("src/explore_jesus.ipynb") as f:
    nb_b = nbformat.read(f, as_version=4)

with open("src/explore_maria.ipynb") as f:
    nb_a = nbformat.read(f, as_version=4)

# Añadir celdas de B a A
nb_a.cells.extend(nb_b.cells)
nb_a.cells.extend(nb_c.cells)

# Guardar el notebook combinado
with open("src/explore.ipynb", "w") as f:
    nbformat.write(nb_a, f)