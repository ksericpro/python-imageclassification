/* eslint-disable react/prop-types */
import React, { Component } from "react";

export class Instructions extends Component {
  constructor(props) {
    super(props);
    this.state = {author: props.author,};
  }

  render() {
    return(
      <p>Author::{this.state.author}</p>
    );
  }

}