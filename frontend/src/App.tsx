import { useState } from 'react';
import { LiveKitRoom, VideoConference } from '@livekit/components-react';
import '@livekit/components-styles';
import './App.css';

function App() {
  const [token, setToken] = useState<string>('');
  const [url, setUrl] = useState<string>('');
  const [participantName, setParticipantName] = useState<string>('');
  const [connected, setConnected] = useState(false);

  const connect = async () => {
    if (!participantName) {
      alert('Participant name is required');
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/api/get-token?participant=${encodeURIComponent(participantName)}`);
      const data = await response.json();
      setToken(data.token);
      setUrl(data.url);
      setConnected(true);
    } catch (e) {
      console.error(e);
      alert('Failed to get token. Please check console for details.');
    }
  };

  const onConnected = () => {
    setConnected(true);
    console.log('Connected to room');
  }

  const onDisconnected = () => {
    setConnected(false);
    setToken('');
    console.log('Disconnected from room');
  };

  const onError = (error: Error) => {
    console.error('Error:', error);
  };

  return (
    <div className="container" data-lk-theme="default">
      <h1>E2E Test Agent</h1>
      
      {!connected ? (
        <div className="card">
          <h2>Join Room</h2>
          <div className="form-group">
            <label htmlFor="participant">Your Name:</label>
            <input
              id="participant"
              type="text"
              value={participantName}
              onChange={(e) => setParticipantName(e.target.value)}
              placeholder="Enter your name"
            />
          </div>
          <button onClick={connect}>Connect</button>
        </div>
      ) : (
        <>
          <LiveKitRoom
            serverUrl={url}
            token={token}
            connect={true}
            onConnected={onConnected}
            onDisconnected={onDisconnected}
            onError={onError}
            audio={true}
            video={true}
          >
            <VideoConference />
          </LiveKitRoom>
        </>
      )}
    </div>
  );
}

export default App;
