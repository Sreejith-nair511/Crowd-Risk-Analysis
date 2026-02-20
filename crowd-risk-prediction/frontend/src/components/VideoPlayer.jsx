import React, { useRef, useEffect, useState } from 'react';
import {
  Box,
  Slider,
  Typography,
  IconButton,
  Paper,
  useTheme
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  SkipPrevious,
  SkipNext,
  VolumeUp,
  VolumeOff,
  Fullscreen
} from '@mui/icons-material';
import './VideoPlayer.css';

const VideoPlayer = ({ onFrameChange, currentFrame, videoSrc }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  const theme = useTheme();

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateTime = () => setCurrentTime(video.currentTime);
    const updateDuration = () => setDuration(video.duration);

    video.addEventListener('timeupdate', updateTime);
    video.addEventListener('loadedmetadata', updateDuration);

    return () => {
      video.removeEventListener('timeupdate', updateTime);
      video.removeEventListener('loadedmetadata', updateDuration);
    };
  }, []);

  const togglePlayPause = () => {
    const video = videoRef.current;
    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e, newValue) => {
    const video = videoRef.current;
    video.currentTime = newValue;
  };

  const handleVolumeChange = (e, newValue) => {
    const video = videoRef.current;
    video.volume = newValue;
    setVolume(newValue);
    setIsMuted(newValue === 0);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    const newMuted = !isMuted;
    video.muted = newMuted;
    setIsMuted(newMuted);
    if (newMuted) {
      setVolume(0);
    } else {
      setVolume(0.8);
      video.volume = 0.8;
    }
  };

  const handleTimeUpdate = () => {
    const video = videoRef.current;
    if (video) {
      const assumedFPS = 30;
      const frameNumber = Math.floor(video.currentTime * assumedFPS);
      if (onFrameChange) {
        onFrameChange(frameNumber);
      }
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box sx={{ 
      width: '100%', 
      bgcolor: 'background.paper',
      borderRadius: 2,
      overflow: 'hidden',
      boxShadow: 2
    }}>
      <Box sx={{ 
        position: 'relative', 
        bgcolor: '#000',
        aspectRatio: '16/9'
      }}>
        <video
          ref={videoRef}
          src={videoSrc || ''}
          onTimeUpdate={handleTimeUpdate}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'contain'
          }}
          poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='640' height='480' viewBox='0 0 640 480'%3E%3Crect width='640' height='480' fill='%231a2332'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-family='Arial' font-size='20' fill='%23ffffff'%3ECrowd Analysis Video%3C/text%3E%3C/svg%3E"
        />
        <canvas 
          ref={canvasRef} 
          style={{ 
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none'
          }} 
        />
      </Box>

      <Paper 
        sx={{ 
          p: 2, 
          bgcolor: 'background.default',
          borderRadius: 0
        }}
      >
        {/* Progress Bar */}
        <Box sx={{ mb: 2 }}>
          <Slider
            value={currentTime}
            onChange={handleSeek}
            max={duration || 100}
            step={0.1}
            sx={{ 
              color: 'primary.main',
              height: 6,
              '& .MuiSlider-thumb': {
                width: 16,
                height: 16,
                backgroundColor: 'primary.main',
                '&:hover': {
                  boxShadow: '0 0 0 8px rgba(25, 118, 210, 0.16)'
                }
              }
            }}
          />
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            mt: 0.5
          }}>
            <Typography variant="caption" color="text.secondary">
              {formatTime(currentTime)}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {formatTime(duration)}
            </Typography>
          </Box>
        </Box>

        {/* Controls */}
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between'
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <IconButton 
              onClick={togglePlayPause}
              sx={{ 
                bgcolor: 'primary.main',
                color: 'white',
                '&:hover': {
                  bgcolor: 'primary.dark'
                }
              }}
            >
              {isPlaying ? <Pause /> : <PlayArrow />}
            </IconButton>
            
            <IconButton size="small">
              <SkipPrevious />
            </IconButton>
            
            <IconButton size="small">
              <SkipNext />
            </IconButton>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <IconButton onClick={toggleMute} size="small">
              {isMuted ? <VolumeOff /> : <VolumeUp />}
            </IconButton>
            
            <Box sx={{ width: 100, display: 'flex', alignItems: 'center' }}>
              <Slider
                value={volume}
                onChange={handleVolumeChange}
                max={1}
                step={0.01}
                sx={{ 
                  width: 80,
                  color: 'primary.main',
                  height: 4
                }}
              />
            </Box>
            
            <IconButton size="small">
              <Fullscreen />
            </IconButton>
          </Box>
        </Box>

        {/* Frame Information */}
        <Box sx={{ 
          mt: 2, 
          p: 1, 
          bgcolor: 'background.paper',
          borderRadius: 1
        }}>
          <Typography variant="body2" color="text.secondary">
            Current Frame: <strong>{currentFrame || 0}</strong>
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default VideoPlayer;