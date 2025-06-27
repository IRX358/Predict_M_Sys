# Predict_M_Sys
 An Predictive Maintanence System
 <br>
Uses deep learning
to predict equipment failures from vibration data. It converts signals into spectrograms, processes them with a CNN model ,and delivers accurate predictions. With a Flask-powered web interface, it helps industries reduce downtime , optimize maintenance ,and improve efficiency through early failure detection.
<br>
<h2>Instructions to use</h2>
<div id="instructions-content">
        <ul>
            <li>📂 Upload a PNG or JPG file</li>
            <li>⏳ Wait for the prediction result</li>
            <li>✅ Check result or use Try Again if needed</li>
            <li>📊 Visit <a href="/metrics">Metrics</a> for model performance</li>
        </ul>
    </div>
   <br>
   <h2>
   Model Metrics : 
   </h2>
        <table>
            <thead>
                <tr class="table-info" >
                    <th class="table-info"  scope="col"># </th>
                    <th class="table-info"  scope="col">Metrics </th>
                    <th class="table-info"  scope="col">Values </th>
                </tr>
            </thead>
            <tbody>
                <tr class="table-info" >
                    <th class="table-info"  scope="row">1</th>
                    <td>Accuracy: </td>
                    <td>{{met_data.accuracy}}</td>
                </tr>
                <tr class="table-info" >
                    <th class="table-info"  scope="row">2</th>
      <td>Precision: </td>
      <td>{{met_data.precision}}</td>
    </tr>
    <tr class="table-info" >
        <th scope="row">3</th>
        <td>Recall: </td>
        <td>{{met_data.recall}}</td>
    </tr>
    <tr class="table-info" >
      <th class="table-info"  scope="row">4</th>
      <td>Score: </td>
      <td>{{met_data.score}}</td>
    </tr>
  </tbody>
</table>
