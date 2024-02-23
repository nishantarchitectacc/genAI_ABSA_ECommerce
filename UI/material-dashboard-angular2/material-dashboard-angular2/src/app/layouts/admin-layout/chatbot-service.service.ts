import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ChatbotServiceService {

  constructor(
    
    private http: HttpClient
  ) { }

  getChatbotData(feedbackInput) {
    const headerDict = {
      'Accept': 'application/json',
      'Access-Control-Allow-Headers': '*',
    }
    
    const requestOptions = {                                                                                                                                                                                 
      headers: new HttpHeaders(headerDict), 
    };

    const body = {value: feedbackInput};
    
    // return this.http.get(this.heroesUrl, requestOptions)

    return this.http.post('http://localhost:3000/',body, requestOptions);
  }
}
