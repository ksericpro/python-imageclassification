// Filename - App.js

import axios from "axios";
import React, { Component } from "react";
import Button from "react-bootstrap/Button";
//import "bootstrap/dist/css/bootstrap.min.css";
import constants from "./constants";
import { Instructions } from "./components/Instructions";
import { ImageResult } from "./components/ImageResult";

// Bootstrap CSS
import "bootstrap/dist/css/bootstrap.min.css";
// Bootstrap Bundle JS
import "bootstrap/dist/js/bootstrap.bundle.min";

class App extends Component {

	componentDidMount () {
		/*const script = document.createElement("script");
		script.src = "./config.js";
		script.async = true;
		document.body.appendChild(script);*/
	}
	
	state = {
		// Initially, no file is selected
		selectedFile: null,
		//imgs: event.target.files,
		errorMsg: "Choose before Pressing the Upload button",
		result: ""
	};

	// On file select (from the pop up)
	onFileChange = (event) => {
		// Update the state
		this.setState({
			selectedFile: event.target.files[0],
		});
	};

	// On file upload (click the upload button)
	onFileUpload = () => {
		if (this.state.selectedFile) {
			// Create an object of formData
			const formData = new FormData();

			// Update the formData object
			formData.append(
				"file",
				this.state.selectedFile,
				this.state.selectedFile.name
			);

			// Details of the uploaded file
			console.log(this.state.selectedFile);

			// Request made to the backend api
			// Send formData object
			axios.post(constants.server, formData)
				.then((response) => {
					console.log(response);
					this.setResult(JSON.stringify(response.data));
				}).catch((error) => this.setMsg(error));
		} else {
			console.log("null");
			this.setMsg("No file is selected");
		}
	};

	setMsg = (msg) => {
		this.setState({
			errorMsg: msg
		});
	};

	setResult = (msg) => {
		this.setState({
			result: msg
		});
	};

	// File content to be displayed after
	// file upload is complete
	fileData = () => {
		//console.log(this.state.selectedFile);
		if (this.state.selectedFile) {
			return (
				<div>
					<h4>File Details:</h4>
					<ul>
						<li>File Name:{" "}
						{this.state.selectedFile.name}</li>
						<li>File Type:{" "}
						{this.state.selectedFile.type}</li>
						<li>Last Modified:{" "}
						{this.state.selectedFile.lastModifiedDate.toDateString()}</li>
					</ul>

					<p> {this.state.result} </p>
				</div>
			);
		} else {
			return (
				<div>
					<br />
					<small>
						{this.state.errorMsg}
					</small>
				</div>
			);
		}
	};

	render() {
		const author = "Eric See";
		return (
			<div>
				<h1>Dogs + Cats Image Classification using Tensorflow</h1>
				<Instructions author={author}/>
				<div>
					<input
						type="file"
						onChange={this.onFileChange}
					/>

					<Button variant="primary" onClick={this.onFileUpload}>
						Inference
					</Button>
				</div>
				{this.fileData()}
				<div>
					{/* Display all selected images. Inline Rendering */}   
					{this.state.selectedFile != null &&     
						<div>
							<h4>Inference Result:</h4>
							<img style={{ "maxHeight": "100pt" }} src={URL.createObjectURL(this.state.selectedFile)} />
							<ImageResult selectedFile={this.state.selectedFile}/>
						</div>
					}		

				</div>
			</div>
		);
	}
}

export default App;

