// Filename - App.js
import axios from "axios";
import React, { useState } from "react";
import Button from "react-bootstrap/Button";

import constants from "./constants";
import ImageResult from "./components/ImageResult";

// Bootstrap CSS
import "bootstrap/dist/css/bootstrap.min.css";
// Bootstrap Bundle JS
import "bootstrap/dist/js/bootstrap.bundle.min";


function App() {
	// Declare a new state variable, which we'll call "count"
	//const [count, setCount] = useState(0);
	//const [sharedState, setSharedState] = useState("initial value");
	const [selectedFile, setSelectedFile] = useState(null);
	const [errorMsg, setErrorMsg] = useState("");
	const [classification_result, setClassificationResult] = useState("");

	// On file select (from the pop up)
	const onFileChange = (event) => {
		// Update the state
		setSelectedFile(event.target.files[0]);
		setErrorMsg("");
		setClassificationResult("");
	};

	// On file upload (click the upload button)
	const onFileUpload = () => {
		if (selectedFile) {
			// Create an object of formData
			const formData = new FormData();

			// Update the formData object
			formData.append(
				"file",
				selectedFile,
				selectedFile.name
			);

			// Details of the uploaded file
			console.log(selectedFile);

			// Request made to the backend api
			// Send formData object
			axios.post(constants.server, formData)
				.then((response) => {
					console.log(response);
					setClassificationResult(response.data["result"]);
				}).catch((error) => this.setMsg(error));
		} else {
			this.setMsg("No file is selected");
		}
	};

	// File content to be displayed after
	// file upload is complete
	const fileData = () => {
		//console.log(this.state.selectedFile);
		if (selectedFile) {
			return (
				<div>
					<h4>File Details:</h4>
					<p><small>
						File Name:{" "}
						{selectedFile.name}
						<br/>
						File Type:{" "}
						{selectedFile.type}
						<br/>
						Last Modified:{" "}
						{selectedFile.lastModifiedDate.toDateString()}
						</small>
					</p>
				</div>
			);
		} else {
			return (
				<div>
					<br />
					<small>
						{errorMsg}
					</small>
				</div>
			);
		}
	};

    return (
		<div>
			<h3>Image Classification using Tensorflow</h3>
			{/*	<Instructions author={author}/>*/}
			<div>
				<input
					type="file"
					onChange={onFileChange}
				/>

				<Button variant="primary" onClick={onFileUpload}>
					Inference
				</Button>
			</div>
			{fileData()}
			<div>	
				<ImageResult sharedState={classification_result} setSharedState={setClassificationResult} selectedFile={selectedFile} />
			</div>
		</div>
    );
}

export default App;