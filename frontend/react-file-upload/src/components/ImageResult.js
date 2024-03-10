/* eslint-disable react/prop-types */
import React from "react";

function ImageResult({ sharedState, setSharedState, selectedFile }) {
	return (
		<div>
			{/* Display all selected images. Inline Rendering */}   
			{selectedFile != null &&     
				<div>
					<img style={{ "maxHeight": "200pt", "border-radius":"10pt" }} src={URL.createObjectURL(selectedFile)} />					
				</div>
			}
			<button onClick={() => setSharedState("changed new value from A")}>
				Result: {sharedState}
			</button>
		</div>
	);
}

export default ImageResult;