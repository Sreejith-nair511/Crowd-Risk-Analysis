import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
import math

class ScenarioGenerator:
    """
    Generator for synthetic crowd scenarios with varying levels of instability
    """
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Initialize random seed for reproducible scenarios
        self.rng = np.random.RandomState(42)

    def generate_bidirectional_flow(self, 
                                  duration_frames: int, 
                                  person_count: int = 50,
                                  flow_directions: Tuple[float, float] = (0, np.pi)) -> List[np.ndarray]:
        """
        Generate bidirectional crowd flow scenario
        Args:
            duration_frames: Number of frames to generate
            person_count: Number of people in each flow direction
            flow_directions: Directions for the two flows (in radians)
        Returns:
            List of frames showing bidirectional flow
        """
        frames = []
        
        # Define the two groups of people
        group1_positions = self._initialize_positions(person_count, self.width, self.height)
        group2_positions = self._initialize_positions(person_count, self.width, self.height)
        
        # Define velocities for each group
        group1_velocities = self._initialize_velocities(person_count, flow_directions[0])
        group2_velocities = self._initialize_velocities(person_count, flow_directions[1])
        
        for frame_idx in range(duration_frames):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Update positions based on velocities
            group1_positions = self._update_positions(group1_positions, group1_velocities, frame_idx)
            group2_positions = self._update_positions(group2_positions, group2_velocities, frame_idx)
            
            # Draw people on the frame
            frame = self._draw_people(frame, group1_positions, color=(255, 0, 0))  # Red
            frame = self._draw_people(frame, group2_positions, color=(0, 0, 255))  # Blue
            
            frames.append(frame)
        
        return frames

    def simulate_bottleneck_compression(self, 
                                     base_scenario: List[np.ndarray], 
                                     bottleneck_position: Tuple[int, int] = None,
                                     bottleneck_width: int = 100,
                                     compression_factor: float = 2.0) -> List[np.ndarray]:
        """
        Simulate bottleneck compression in crowd flow
        Args:
            base_scenario: Base scenario frames to apply bottleneck to
            bottleneck_position: Center position of the bottleneck (x, y)
            bottleneck_width: Width of the bottleneck region
            compression_factor: Factor by which to compress the crowd
        Returns:
            Frames with bottleneck effect applied
        """
        if bottleneck_position is None:
            bottleneck_position = (self.width // 2, self.height // 2)
        
        bottleneck_x, bottleneck_y = bottleneck_position
        bottleneck_radius = bottleneck_width // 2
        
        compressed_frames = []
        
        for frame in base_scenario:
            compressed_frame = frame.copy()
            
            # Find pixels that are in the bottleneck region
            y_coords, x_coords = np.ogrid[:self.height, :self.width]
            dist_from_center = np.sqrt((x_coords - bottleneck_x)**2 + (y_coords - bottleneck_y)**2)
            
            # Apply compression effect based on distance from bottleneck center
            compression_mask = dist_from_center <= bottleneck_radius
            compression_intensity = np.where(compression_mask, 
                                           compression_factor, 1.0)
            
            # For simplicity, we'll increase the density by duplicating people
            # in the bottleneck region
            for y in range(self.height):
                for x in range(self.width):
                    if compression_mask[y, x]:
                        # Increase density in bottleneck area
                        if np.sum(frame[y, x]) > 0:  # If there's a person
                            # Add more "people" (pixels) around this position
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < self.height and 0 <= nx < self.width:
                                        compressed_frame[ny, nx] = frame[y, x]
            
            compressed_frames.append(compressed_frame)
        
        return compressed_frames

    def generate_stress_test_video(self, 
                                 scenario_params: Dict) -> List[np.ndarray]:
        """
        Generate stress-test videos with various instability patterns
        Args:
            scenario_params: Dictionary of parameters for scenario generation
        Returns:
            List of frames for stress test video
        """
        # Default parameters
        params = {
            'duration_frames': 300,
            'person_count': 50,
            'bottleneck_position': (self.width // 2, self.height // 2),
            'bottleneck_width': 100,
            'compression_factor': 2.0,
            'conflict_probability': 0.1,  # Probability of conflict per frame
            'panic_probability': 0.05,    # Probability of panic spread per frame
            'density_increase_rate': 0.01  # Rate at which density increases
        }
        
        # Update with provided parameters
        params.update(scenario_params)
        
        # Generate base bidirectional flow
        base_frames = self.generate_bidirectional_flow(
            duration_frames=params['duration_frames'],
            person_count=params['person_count']
        )
        
        # Apply bottleneck compression
        bottleneck_frames = self.simulate_bottleneck_compression(
            base_scenario=base_frames,
            bottleneck_position=params['bottleneck_position'],
            bottleneck_width=params['bottleneck_width'],
            compression_factor=params['compression_factor']
        )
        
        # Add stress elements to frames
        stress_frames = []
        current_density_factor = 1.0
        
        for i, frame in enumerate(bottleneck_frames):
            stress_frame = frame.copy()
            
            # Gradually increase density
            current_density_factor += params['density_increase_rate']
            
            # Occasionally add conflicts or panic-inducing events
            if self.rng.random() < params['conflict_probability']:
                stress_frame = self._add_conflict_event(stress_frame, i)
            
            if self.rng.random() < params['panic_probability']:
                stress_frame = self._spread_panic(stress_frame, i)
            
            stress_frames.append(stress_frame)
        
        return stress_frames

    def generate_crowd_density_scenario(self, 
                                      duration_frames: int = 300,
                                      min_people: int = 20,
                                      max_people: int = 100,
                                      density_pattern: str = 'increasing') -> List[np.ndarray]:
        """
        Generate scenario with varying crowd density
        Args:
            duration_frames: Number of frames to generate
            min_people: Minimum number of people
            max_people: Maximum number of people
            density_pattern: Pattern of density change ('increasing', 'decreasing', 'oscillating')
        Returns:
            List of frames with varying crowd density
        """
        frames = []
        
        for frame_idx in range(duration_frames):
            # Calculate number of people based on pattern
            if density_pattern == 'increasing':
                t = frame_idx / duration_frames
                n_people = int(min_people + (max_people - min_people) * t)
            elif density_pattern == 'decreasing':
                t = frame_idx / duration_frames
                n_people = int(max_people - (max_people - min_people) * t)
            elif density_pattern == 'oscillating':
                t = frame_idx / duration_frames
                amplitude = (max_people - min_people) / 2
                center = (max_people + min_people) / 2
                n_people = int(center + amplitude * np.sin(4 * np.pi * t))
                n_people = max(min_people, min(n_people, max_people))
            else:
                n_people = min_people
            
            # Generate positions for this frame
            positions = self._initialize_positions(n_people, self.width, self.height)
            
            # Create frame with people
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame = self._draw_people(frame, positions)
            
            frames.append(frame)
        
        return frames

    def generate_instability_scenario(self,
                                   duration_frames: int = 300,
                                   base_person_count: int = 50,
                                   instability_frequency: float = 0.1) -> List[np.ndarray]:
        """
        Generate scenario with periodic instability events
        Args:
            duration_frames: Number of frames to generate
            base_person_count: Base number of people
            instability_frequency: Frequency of instability events (0.0 to 1.0)
        Returns:
            List of frames with periodic instability
        """
        frames = []
        
        for frame_idx in range(duration_frames):
            # Base scenario
            positions = self._initialize_positions(base_person_count, self.width, self.height)
            
            # Add instability events periodically
            if self.rng.random() < instability_frequency:
                # Add chaotic movement
                positions = self._add_chaotic_movement(positions, intensity=0.5)
            
            # Create frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame = self._draw_people(frame, positions)
            
            frames.append(frame)
        
        return frames

    def _initialize_positions(self, count: int, width: int, height: int) -> np.ndarray:
        """Initialize random positions for people"""
        positions = np.zeros((count, 2))
        positions[:, 0] = self.rng.uniform(0, width, count)  # x coordinates
        positions[:, 1] = self.rng.uniform(0, height, count)  # y coordinates
        return positions

    def _initialize_velocities(self, count: int, direction: float, speed_range: Tuple[float, float] = (1.0, 3.0)) -> np.ndarray:
        """Initialize velocities for people in a given direction"""
        velocities = np.zeros((count, 2))
        speed = self.rng.uniform(speed_range[0], speed_range[1], count)
        
        velocities[:, 0] = speed * np.cos(direction)  # x component
        velocities[:, 1] = speed * np.sin(direction)  # y component
        
        return velocities

    def _update_positions(self, positions: np.ndarray, velocities: np.ndarray, frame_idx: int) -> np.ndarray:
        """Update positions based on velocities with boundary handling"""
        new_positions = positions + velocities
        
        # Handle boundary collisions (simple bounce)
        for i in range(len(new_positions)):
            if new_positions[i, 0] < 0 or new_positions[i, 0] >= self.width:
                velocities[i, 0] *= -1  # Reverse x velocity
                new_positions[i, 0] = np.clip(new_positions[i, 0], 0, self.width - 1)
            
            if new_positions[i, 1] < 0 or new_positions[i, 1] >= self.height:
                velocities[i, 1] *= -1  # Reverse y velocity
                new_positions[i, 1] = np.clip(new_positions[i, 1], 0, self.height - 1)
        
        return new_positions

    def _draw_people(self, frame: np.ndarray, positions: np.ndarray, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Draw people on the frame"""
        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.width and 0 <= y < self.height:
                # Draw a small circle for each person
                cv2.circle(frame, (x, y), radius=3, color=color, thickness=-1)
        
        return frame

    def _add_conflict_event(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Add a conflict event to the frame"""
        # Select a random region
        center_x = self.rng.randint(50, self.width - 50)
        center_y = self.rng.randint(50, self.height - 50)
        
        # Mark the conflict area (red region)
        cv2.circle(frame, (center_x, center_y), radius=30, color=(0, 0, 255), thickness=2)
        
        return frame

    def _spread_panic(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Simulate panic spreading effect"""
        # Find people near the center and make them move chaotically
        center_x, center_y = self.width // 2, self.height // 2
        
        # Add visual indicators of panic (flashing)
        if frame_idx % 5 < 2:  # Flash every 5 frames
            cv2.rectangle(frame, (0, 0), (self.width, self.height), (0, 0, 255), -1)
        
        return frame

    def _add_chaotic_movement(self, positions: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Add chaotic movement to positions"""
        chaotic_offsets = self.rng.normal(0, intensity, positions.shape)
        return positions + chaotic_offsets

    def save_video(self, frames: List[np.ndarray], output_path: str, fps: int = None):
        """
        Save frames as a video file
        Args:
            frames: List of frames to save
            output_path: Path to save the video
            fps: Frames per second (uses instance fps if not provided)
        """
        if fps is None:
            fps = self.fps
        
        if not frames:
            raise ValueError("No frames provided to save")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Ensure frame is the right type and size
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            out.write(frame)
        
        out.release()

    def generate_scenario_dataset(self, 
                                num_scenarios: int = 10,
                                scenario_types: List[str] = None) -> List[Dict]:
        """
        Generate a dataset of different scenarios
        Args:
            num_scenarios: Number of scenarios to generate
            scenario_types: List of scenario types to include
        Returns:
            List of scenario dictionaries
        """
        if scenario_types is None:
            scenario_types = ['bidirectional', 'bottleneck', 'density_varying', 'instability']
        
        scenarios = []
        
        for i in range(num_scenarios):
            scenario_type = scenario_types[i % len(scenario_types)]
            
            if scenario_type == 'bidirectional':
                frames = self.generate_bidirectional_flow(
                    duration_frames=self.rng.randint(150, 300),
                    person_count=self.rng.randint(30, 70)
                )
                label = 'bidirectional_flow'
                
            elif scenario_type == 'bottleneck':
                base_frames = self.generate_bidirectional_flow(
                    duration_frames=self.rng.randint(150, 300),
                    person_count=self.rng.randint(40, 80)
                )
                frames = self.simulate_bottleneck_compression(
                    base_scenario=base_frames,
                    bottleneck_position=(
                        self.rng.randint(100, self.width - 100),
                        self.rng.randint(100, self.height - 100)
                    ),
                    compression_factor=self.rng.uniform(1.5, 3.0)
                )
                label = 'bottleneck_compression'
                
            elif scenario_type == 'density_varying':
                frames = self.generate_crowd_density_scenario(
                    duration_frames=self.rng.randint(200, 400),
                    min_people=self.rng.randint(20, 50),
                    max_people=self.rng.randint(60, 120),
                    density_pattern=self.rng.choice(['increasing', 'decreasing', 'oscillating'])
                )
                label = 'density_varying'
                
            elif scenario_type == 'instability':
                frames = self.generate_instability_scenario(
                    duration_frames=self.rng.randint(200, 400),
                    base_person_count=self.rng.randint(40, 80),
                    instability_frequency=self.rng.uniform(0.05, 0.2)
                )
                label = 'instability'
            
            scenarios.append({
                'frames': frames,
                'label': label,
                'metadata': {
                    'scenario_type': scenario_type,
                    'frame_count': len(frames),
                    'generated_at': i
                }
            })
        
        return scenarios


def generate_sample_scenarios(output_dir: str = "data/synthetic"):
    """
    Generate sample scenarios and save them
    Args:
        output_dir: Directory to save the generated scenarios
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    generator = ScenarioGenerator(width=640, height=480)
    
    # Generate different types of scenarios
    scenarios = [
        ("bidirectional_flow.mp4", lambda: generator.generate_bidirectional_flow(300, 40)),
        ("bottleneck_compression.mp4", lambda: generator.simulate_bottleneck_compression(
            generator.generate_bidirectional_flow(300, 50))),
        ("density_increasing.mp4", lambda: generator.generate_crowd_density_scenario(
            300, 20, 100, 'increasing')),
        ("stress_test.mp4", lambda: generator.generate_stress_test_video({}))
    ]
    
    for filename, gen_func in scenarios:
        filepath = os.path.join(output_dir, filename)
        frames = gen_func()
        generator.save_video(frames, filepath)
        print(f"Generated scenario: {filepath} ({len(frames)} frames)")
    
    print(f"All scenarios saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    generator = ScenarioGenerator()
    
    # Generate a sample scenario
    frames = generator.generate_bidirectional_flow(duration_frames=100, person_count=30)
    
    # Save the video
    generator.save_video(frames, "sample_bidirectional.mp4")
    
    print("Sample bidirectional flow video generated: sample_bidirectional.mp4")