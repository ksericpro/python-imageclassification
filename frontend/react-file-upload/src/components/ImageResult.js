/* eslint-disable react/prop-types */
import React, { Component } from "react";

export class ImageResult extends Component {
  constructor(props) {
    super(props);
   // this.state = {author: props.author,};
  }

  render() {
    return(
      <div>
        <img/>
        <p>Dot or Cat</p>
      </div>
    );
  }

}