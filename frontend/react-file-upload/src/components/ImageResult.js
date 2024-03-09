/* eslint-disable react/prop-types */
import React, { Component } from "react";

export class ImageResult extends Component {
  constructor(props) {
    super(props);
    this.state = {
      result: "Dogs",
      selectedFile: null
    };
  }

  render() {
    return(
      <div>
        <img/>
        <p>It is a {this.state.result}</p>
      </div>
    );
  }

}