import {Component, Inject} from '@angular/core';
import {Dialog, DialogRef, DIALOG_DATA, DialogModule} from '@angular/cdk/dialog';
import {FormsModule} from '@angular/forms';

export interface DialogData {
  animal: string;
  name: string;
}

/**
 * @title CDK Dialog Overview
 */
@Component({
  selector: 'app-pop-up',
  templateUrl: './pop-up.component.html',
  styleUrls: ['./pop-up.component.scss'],
})
export class PopUpComponent {
  animal: string | undefined;
  name: string;

  constructor(public dialog: Dialog, @Inject(DIALOG_DATA) public data: DialogData) {
    console.log("dialog", this.data);
  }

  // openDialog(): void {
  //   const dialogRef = this.dialog.open<string>(PopUpComponent, {
  //     width: '250px',
  //     data: {name: this.name, animal: this.animal},
  //   });

  //   dialogRef.closed.subscribe(result => {
  //     console.log('The dialog was closed');
  //     this.animal = result;
  //   });
  // }

  close() {
    this.dialog.closeAll();
  }
}

