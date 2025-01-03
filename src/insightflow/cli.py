import click
import torch
from insightflow.train import dataset_to_model

@click.group()
def cli():
    """InsightFlow CLI Tool."""
    pass


@cli.command()
@click.option('--data_path', help='Path of csv file containing data')
@click.option('--num_epochs', default=10, show_default=True, help='Number of training epochs')
@click.option('--batch_size', default=1, show_default=True, help='Number of elements per batch')
@click.option('--learning_rate', default=0.001, show_default=True, help='Learning rate per iteration')
@click.option('--device', default='cpu', show_default=True, help='device to compute on (GPU/CPU)')
def train(data_path, num_epochs, batch_size, learning_rate, device):
    """
    Train the InsightFlow model.
    """
    # Call the existing function with CLI-provided arguments
    trainer = dataset_to_model(
        data_path=data_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )
    
    click.echo("Training completed successfully.")