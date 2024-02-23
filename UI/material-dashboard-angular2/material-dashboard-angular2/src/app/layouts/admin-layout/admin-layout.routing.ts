import { Routes } from '@angular/router';

import { HomeComponent } from 'app/components/home/home.component';
import { ProductsComponent } from 'app/components/products/products.component';

export const AdminLayoutRoutes: Routes = [
    { path: 'home',   component: HomeComponent },
    { path: 'products',          component: ProductsComponent }
];
