import React, { useState } from "react";
import axios from "axios";

function App() {
  const [video, setVideo] = useState(null);
  const [streamURL, setStreamURL] = useState("");

  const handleUpload = async () => {
    if (!video) return alert("Please select a video!");

    const formData = new FormData();
    formData.append("video", video);

    try {
      const res = await axios.post("http://localhost:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log("[DEBUG] Uploaded video response:", res.data);

      // Set stream URL
      setStreamURL(`http://localhost:5000/stream/${res.data.video_name}`);
    } catch (err) {
      console.error("[ERROR] Upload failed:", err);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>🎥 Face Recognition</h1>
      <input
        type="file"
        accept="video/*"
        onChange={(e) => setVideo(e.target.files[0])}
      />
      <button onClick={handleUpload} style={{ marginLeft: "10px" }}>
        Upload & Stream
      </button>

      {streamURL && (
        <div style={{ marginTop: "20px" }}>
          <h2>Video Stream:</h2>
          <img src={streamURL} style={{ width: "640px" }} alt="stream" />
        </div>
      )}
    </div>
  );
}

export default App;
