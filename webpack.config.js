const path = require('path');

module.exports = {
	module: {
		rules: [
			{
				test: /\.jsx?$/,
				exclude: /node_modules/,
				use: {
					loader: 'babel-loader',
				}
			},
			{
				test: /\.css$/,
				use: {
					loader: 'css-loader',
				}
			},
			{
				test: /\.(txt|svg|ttf|gif|jpe?g|woff2?|eot)$/,
				use: {
					loader: 'file-loader',
				}
			},

		]
	},
	watch: true,
};
