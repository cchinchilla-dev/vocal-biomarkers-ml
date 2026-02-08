"""Unit tests for __main__ module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestMain:
    """Test suite for main entry point."""

    def test_main_with_default_config(self) -> None:
        """Test main function with default config path."""
        with patch("voice_analysis.__main__.VoiceAnalysisPipeline") as mock_pipeline:
            with patch("voice_analysis.__main__.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                mock_pipeline_instance = MagicMock()
                mock_pipeline.return_value = mock_pipeline_instance

                from voice_analysis.__main__ import main

                with patch.object(sys, "argv", ["voice-analysis"]):
                    main()

                mock_pipeline_instance.run.assert_called_once()

    def test_main_with_custom_config(self, tmp_path: Path) -> None:
        """Test main function with custom config path."""
        config_path = tmp_path / "custom_config.yaml"
        config_path.touch()

        with patch("voice_analysis.__main__.VoiceAnalysisPipeline") as mock_pipeline:
            with patch("voice_analysis.__main__.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                mock_pipeline_instance = MagicMock()
                mock_pipeline.return_value = mock_pipeline_instance

                from voice_analysis.__main__ import main

                with patch.object(sys, "argv", ["voice-analysis", "--config", str(config_path)]):
                    main()

                mock_settings.assert_called_once()

    def test_main_handles_keyboard_interrupt(self) -> None:
        """Test that main handles KeyboardInterrupt gracefully."""
        with patch("voice_analysis.__main__.VoiceAnalysisPipeline") as mock_pipeline:
            with patch("voice_analysis.__main__.get_settings") as mock_settings:
                mock_settings.return_value = MagicMock()
                mock_pipeline_instance = MagicMock()
                mock_pipeline_instance.run.side_effect = KeyboardInterrupt()
                mock_pipeline.return_value = mock_pipeline_instance

                from voice_analysis.__main__ import main

                with patch.object(sys, "argv", ["voice-analysis"]):
                    with pytest.raises(SystemExit):
                        main()

    def test_main_handles_file_not_found(self) -> None:
        """Test main handles missing config file."""
        with patch("voice_analysis.__main__.get_settings") as mock_settings:
            mock_settings.side_effect = FileNotFoundError("Config not found")

            from voice_analysis.__main__ import main

            with patch.object(sys, "argv", ["voice-analysis", "--config", "nonexistent.yaml"]):
                with pytest.raises(SystemExit):
                    main()


class TestCommandLineInterface:
    """Test suite for CLI argument parsing."""

    def test_parse_args_default(self) -> None:
        """Test argument parsing with defaults."""
        with patch.object(sys, "argv", ["voice-analysis"]):
            from voice_analysis.__main__ import parse_args

            args = parse_args()
            assert args.config == Path("config.yaml")

    def test_parse_args_custom_config(self) -> None:
        """Test argument parsing with custom config."""
        with patch.object(sys, "argv", ["voice-analysis", "--config", "custom.yaml"]):
            from voice_analysis.__main__ import parse_args

            args = parse_args()
            assert args.config == Path("custom.yaml")

    def test_parse_args_verbose(self) -> None:
        """Test argument parsing with verbose flag."""
        with patch.object(sys, "argv", ["voice-analysis", "-v"]):
            from voice_analysis.__main__ import parse_args

            args = parse_args()
            assert args.verbose is True
