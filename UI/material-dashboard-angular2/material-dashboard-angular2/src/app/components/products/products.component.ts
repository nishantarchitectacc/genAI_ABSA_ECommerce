import { animate, state, style, transition, trigger } from '@angular/animations';
import { Dialog } from '@angular/cdk/dialog';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { PopUpComponent } from 'app/pop-up/pop-up.component';


const fadeInOut = trigger('fadeInOut', [
  state(
    'in',
    style({
      opacity: 1,
    })
  ),
  transition('void => *', [style({ opacity: 0 }), animate('1s ease-out')]),
  transition('* => void', [animate('1s ease-out'), style({ opacity: 0 })]),
]);

@Component({
  selector: 'app-products',
  templateUrl: './products.component.html',
  styleUrls: ['./products.component.scss'],
  animations: [fadeInOut]
})
export class ProductsComponent implements OnInit {
  showProductDetails = false;
  showChatBot = false;
  positiveFeedbacks = [
    "Battery life", "Setup", "Noise cancelling", "Sound quality", "Ease of use", "Comfortable to wear", "Fast charging", "Brilliant fit", "Good quality", "Excellent product for calls and music"
  ];
  negativeFeedback = [
    "Poor sound quality", "Design flaws", "Bluetooth connectivity issues", "Limited battery life", "Noise cancelation not as good as expected", "Build quality is not up to the mark", "Not worth the price", "Lack of features", "Difficulty in pairing with devices", "Disappointing overall experience"
  ];
  actionItems = [
    "Enhance audio quality through comprehensive analysis, hardware upgrades, EQ adjustments, and noise reduction features.",
    "Improve JBL Live Pro 2 TWS user experience by addressing app crashes, Bluetooth range limitations, and latency issues through software optimizations and hardware enhancements.",
    "Enhance battery life for JBL Live Pro 2 TWS by increasing battery capacity, optimizing power management, and releasing software updates to improve efficiency and reduce power consumption",
    "Enhance user experience with JBL Live Pro 2 TWS by improving noise reduction, customization options, noise isolation, and overall sound quality for a more immersive and tailored listening experience.",
    "Enhance JBL Live Pro 2 TWS by prioritizing lightweight, durable materials, extending battery life, advancing noise cancellation, enabling fast pairing with Bluetooth 5.0, and ensuring clear voice transmission with high-quality microphones.",
    "Establish a competitive pricing strategy by analyzing cost structure, market pricing, value proposition, customer perception, and desired profit margins.",
    "Conduct market analysis to evaluate competitor features and understand user needs thoroughly, aiming to identify valuable additions that can create a wow factor",
    "Enhance user experience by ensuring compatibility with various devices, improving Bluetooth connectivity, and providing clear instructions for seamless pairing of JBL headphones.",
    "Enhance JBL headphones by improving noise cancellation against voices and higher frequencies, redesigning for better comfort with adjustable wingtips, and adding quick charging for instant boosts to the 10-hour battery life, ultimately boosting customer satisfaction.",
  ];
  constructor(
    public dialog: Dialog
  ) { }

  ngOnInit(): void {
    // const headerDict = {
    //   'Accept': 'application/json',
    //   'Access-Control-Allow-Headers': '*',
    //   'Access-Control-Allow-Origin': '*'
    // }
    
    // const requestOptions = {                                                                                                                                                                                 
    //   headers: new HttpHeaders(headerDict), 
    // };
    
    // // return this.http.get(this.heroesUrl, requestOptions)

    // this.http.get('http://localhost:3000/', requestOptions).subscribe((val) => console.log(val));
  }

  productSelected() {
    this.showProductDetails = true;
  }

  navigateToQueries() {
    console.log('clicked');
    this.showChatBot = true;
  }

  previousPage($event) {
    this.showChatBot = !$event;
    console.log($event);
  }

  showDialog(str) {
    console.log('str', str);
    this.dialog.open<string>(PopUpComponent, {minWidth: '300px',data: str});
  }

}
