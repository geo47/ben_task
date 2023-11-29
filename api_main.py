from flask import Flask, render_template, request

from src.chain_pipeline import ChainPipeline

app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/refresh")
def refresh():
    # response = action_api_class.get_started()
    # print(response)
    return "refresh"


@app.route("/get")
def get_bot_response():
    query = str(request.args.get('query'))
    result = chain_pipeline.run(query)
    return result


if __name__ == "__main__":
    chain_pipeline = ChainPipeline()
    app.run()
