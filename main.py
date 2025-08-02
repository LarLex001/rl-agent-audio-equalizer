# Copyright 2025 Oleksii Larionov
# Licensed under the Apache License, Version 2.0
# See LICENSE file for more information.

import numpy as np
import os
import logging
from datetime import datetime
from src.utils.feature_extraction import extract_feature_vector
from src.agent.eq_action_space import EQActionSpace 
from src.agent.dqn_eq_agent import EQAgent
from src.training.rl_agent_train import train_eq_agent
from src.training.rl_agent_test import test_eq_agent, test_single_genre


def setup_logging(log_dir="rl-agent-audio-equalizer/logs"):
    """
    Setup logging configuration with timestamp and file output.
    
    Parameters:
      log_dir: directory to save log files
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"eq_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():

    logger = setup_logging()
    logger.info("Starting EQ Agent Training")

    DATASET_PATH = "rl-agent-audio-equalizer/data/GTZAN" 
    SAVE_PATH = "rl-agent-audio-equalizer/result"  
    TEST_PATH = "rl-agent-audio-equalizer/data/GTZAN"
    
    # Training hyperparameters
    NUM_EPISODES = 200
    MAX_STEPS = 30
    PATIENCE = 120
    LOG_INTERVAL = 10
    HIDDEN_SIZE = 512
    SAMPLE_RATE = 22050
    
    # Feature configuration
    feature_types = [
        'mel_spectrogram', 
        'mfcc', 
        'chroma', 
        'spectral_contrast', 
        'spectral_centroid', 
        'spectral_flatness'
    ]
    
    try:
        # Initialize action space
        logger.info("Initializing action space...")
        action_space = EQActionSpace()
        action_space_sizes = action_space.get_action_space_size()
        logger.info(f"Action space sizes: {action_space_sizes}")
        
        # Calculate feature vector size
        logger.info("Calculating feature vector size...")
        dummy_audio1 = np.random.randn(SAMPLE_RATE) 
        dummy_audio2 = np.random.randn(SAMPLE_RATE)
        feature_vector = extract_feature_vector(
            dummy_audio1, 
            dummy_audio2, 
            SAMPLE_RATE, 
            feature_types
        )
        feature_size = len(feature_vector)
        logger.info(f"Feature vector size: {feature_size}")
        
        # Initialize agent
        logger.info("Initializing EQ Agent...")
        agent = EQAgent(
            feature_size, 
            action_space_sizes, 
            hidden_size=HIDDEN_SIZE
        )
        
        # Training phase
        logger.info("Starting training phase...")
        training_stats = train_eq_agent(
            agent, 
            action_space, 
            DATASET_PATH, 
            num_episodes=NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            early_stopping_patience=PATIENCE,
            save_path=os.path.join(SAVE_PATH, "train_results"),
            log_interval=LOG_INTERVAL,
            feature_types=feature_types
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Training statistics: {training_stats}")
        
        # Testing phase
        logger.info("Starting testing phase...")
        test_stats = test_eq_agent(
            agent, 
            action_space, 
            TEST_PATH, 
            genres=None, 
            feature_types=feature_types, 
            save_path=os.path.join(SAVE_PATH, "test_results"),
            save_processed_audio=True,
            save_visualizations=True,
            create_dataset=True
        )

        # logger.info("Starting testing phase for separate genre...")
        # test_stats = test_single_genre(
        #     agent, 
        #     action_space, 
        #     TEST_PATH, 
        #     genre="blues", 
        #     feature_types=feature_types, 
        #     save_path=os.path.join(SAVE_PATH, "test_results")
        # )
        
        logger.info("Testing completed successfully!")
        logger.info(f"Test statistics: {test_stats}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
