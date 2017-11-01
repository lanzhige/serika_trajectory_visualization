/**
 * Created by idmv61 on 5/23/2017.
 * This class is used to define POI click events
 */

var ClickEventHandler = function(map) {
    this.map = map;
    /*this.directionsService = new google.maps.DirectionsService;
    this.directionsDisplay = new google.maps.DirectionsRenderer;
    this.directionsDisplay.setMap(map);*/
    this.placesService = new google.maps.places.PlacesService(map);
    this.infowindow = new google.maps.InfoWindow;
    this.infowindowContent = document.getElementById('POI-info');
    this.infowindow.setContent(this.infowindowContent);

    // Listen for clicks on the map.
    this.map.addListener('click', this.handleClick.bind(this));
};

ClickEventHandler.prototype.handleClick = function(event) {
    // If the event has a placeId, use it.
    if (event.placeId) {
        console.log('You clicked on place:' + event.placeId);
        console.log('You clicked on: ' + event.latLng);
        // Calling e.stop() on the event prevents the default info window from
        // showing.
        // If you call stop here when there is no placeId you will prevent some
        // other map click event handlers from receiving the event.
        event.stop();
        //this.calculateAndDisplayRoute(event.placeId);
        console.log(event);
        this.event = event;
        this.getPlaceInformation(event.placeId);
    } else {
    }
};

ClickEventHandler.prototype.calculateAndDisplayRoute = function(placeId) {
    /*var me = this;
    this.directionsService.route({
        origin: this.origin,
        destination: {placeId: placeId},
        travelMode: 'WALKING'
    }, function(response, status) {
        if (status === 'OK') {
            me.directionsDisplay.setDirections(response);
        } else {
            window.alert('Directions request failed due to ' + status);
        }
    });*/
};

ClickEventHandler.prototype.getPlaceInformation = function(placeId) {
    var me = this;
    this.placesService.getDetails({placeId: placeId}, function(place, status) {
        if (status === 'OK') {
            me.infowindow.close();
            me.infowindow.setPosition(place.geometry.location);
            me.infowindowContent.children['place-icon'].src = place.icon;
            me.infowindowContent.children['place-name'].textContent = place.name;
            //me.infowindowContent.children['place-id'].textContent = place.place_id;
            me.infowindowContent.children['place-address'].textContent =
                place.formatted_address;
            me.infowindow.open(me.map);
            me.setButton(place);
            me.placeInfo = place;
        }
    });
};

ClickEventHandler.prototype.setButton = function(place) {
    console.log(this.infowindowContent.children['POI-add'].textContent);
    if (POI_id_list.indexOf(place.place_id)==-1) this.infowindowContent.children['POI-add'].textContent = "add POI";
        else this.infowindowContent.children['POI-add'].textContent = "remove POI";
   /* if (POI_display_list.indexOf(place.place_id)==-1)
            this.infowindowContent.children['POI-display-chord'].textContent = "display";
            else this.infowindowContent.children['POI-display-chord'].textContent = "hide";*/
   if (cnt_geoobj==undefined) $("#POI-display-chord").prop("disabled",true);
        else $("#POI-display-chord").prop("disabled",false);
};