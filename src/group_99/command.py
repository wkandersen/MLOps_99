import typer
from data import load_data as load_data_function


app = typer.Typer()
load_app = typer.Typer()
app.add_typer(load_app, name="data")

# Load the dataset
@load_app.command()
def load_data():
    data, transform, class_names, dataset_path = load_data_function()
    print(f"Loaded dataset from {dataset_path}")
    print(f"Classes: {class_names}")
    print(f"Number of samples: {len(data)}")
    return data, transform, class_names, dataset_path



if __name__ == "__main__":
    app()
