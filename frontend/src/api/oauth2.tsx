import React, { Component } from 'react';
import { Redirect } from 'react-router';
class OAuth2RedirectHandler extends Component<{ location?: Location }> {
  getUrlParameter(name: any) {
    name = name.replace(/[\[]/, '\\[').replace(/[\]]/, '\\]');
    var regex = new RegExp('[\\?&]' + name + '=([^&#]*)');

    var results = regex.exec(location.search);

    return results === null
      ? ''
      : decodeURIComponent(results[1].replace(/\+/g, ' '));
  }

  render() {
    const token = this.getUrlParameter('token');
    const error = this.getUrlParameter('error');
    console.log('token');
    if (token) {
      localStorage.setItem('isLogin', '1');
      console.log('suc');
      return (
        <Redirect
          to={{
            pathname: '/',
            state: { from: this.props.location },
          }}
        />
      );
    } else {
      console.log('fail');
      return (
        <Redirect
          to={{
            pathname: '/',
            state: {
              from: this.props.location,
              error: error,
            },
          }}
        />
      );
    }
  }
}

export default OAuth2RedirectHandler;
