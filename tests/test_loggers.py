from unittest.mock import patch
from exdark.logging.loggers import ExDarkWandBLogger, ExDarkCSVLogger


@patch("wandb.login")
def test_exdark_wandb_logger_supports_extended_logging(mock_wandb_login):
    logger = ExDarkWandBLogger(token="dummy_token", project="test_project")
    mock_wandb_login.assert_called_once_with(key="dummy_token")
    assert logger.supports_extended_logging is True


def test_exdark_csv_logger_supports_extended_logging():
    logger = ExDarkCSVLogger(save_dir="test_dir")
    assert logger.supports_extended_logging is False
