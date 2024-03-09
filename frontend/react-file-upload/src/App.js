// Filename - App.js

import axios from "axios";
import React, { Component } from "react";
import Button from "react-bootstrap/Button";
import "bootstrap/dist/css/bootstrap.min.css";
import constants from "./constants";
import { Instructions } from "./components/Instructions";
import { ImageResult } from "./components/ImageResult";

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
					<h2>File Details:</h2>
					<p>
						File Name:{" "}
						{this.state.selectedFile.name}
					</p>

					<p>
						File Type:{" "}
						{this.state.selectedFile.type}
					</p>

					<p>
						Last Modified:{" "}
						{this.state.selectedFile.lastModifiedDate.toDateString()}
					</p>

					<p> {this.state.result} </p>
				</div>
			);
		} else {
			return (
				<div>
					<br />
					<h4>
						{this.state.errorMsg}
					</h4>
				</div>
			);
		}
	};

	render() {
		const author = "Eric See";
		return (
			<div>
				<h1>Image Classification</h1>
				<Instructions author={author}/>
				<h3>File Upload using React!</h3>
				<div>
					<input
						type="file"
						onChange={this.onFileChange}
					/>

					<Button variant="primary" onClick={this.onFileUpload}>
						Upload!
					</Button>
				</div>
				{this.fileData()}
				<div>
					<ImageResult/>
				</div>
			</div>
		);
	}
}

export default App;

